
    def _plm_pattern_worker(
        self,
        plm_catalog: str,
        monitor: int,
        interval_ms: float,
        loop_start_index: int,
        loop_stop_index: int,
        hold_pattern_index: int,
    ) -> None:

        message = "PLM loop stopped."

        try:

            phase_patterns = self._plm_phase_patterns

            dynamic_phase_cal = bool(
                phase_patterns is None
                and self._phase_cal_running
                and self._phase_cal_total_patterns > 0
                and len(self._phase_cal_pattern_meta) == self._phase_cal_total_patterns
            )

            if phase_patterns is None and not dynamic_phase_cal:

                raise RuntimeError("No phase patterns loaded.")

            plm = PLM.from_db(plm_catalog)

            expected_h, expected_w = int(plm.shape[0]), int(plm.shape[1])

            try:

                self._plm_pixel_pitch_y_m = float(plm.pitch[0])

                self._plm_pixel_pitch_x_m = float(plm.pitch[1])

            except Exception:

                pass

            if (not dynamic_phase_cal) and (
                phase_patterns.shape[1] != expected_h
                or phase_patterns.shape[2] != expected_w
            ):

                raise ValueError(
                    f"Pattern shape {phase_patterns.shape[1:]} does not match PLM shape {(expected_h, expected_w)} for catalog'{plm_catalog}'."
                )

            phase_levels, buckets, phase_values = get_phase_values_from_calibration(plm)

            n_states = int(len(phase_values))

            if n_states <= 0 or len(phase_levels) != n_states:

                raise RuntimeError("Invalid PLM calibration LUT.")

            interval_s = max(0.01, interval_ms / 1000.0)

            with ImageWindow(fullscreen=True, monitor=monitor) as win:

                frame_idx = 0

                last_update = perf_counter() - interval_s

                if dynamic_phase_cal:

                    loop_start_0_based = 0

                    total_frames = int(self._phase_cal_total_patterns)

                else:

                    available_frames = int(phase_patterns.shape[0])

                    start_idx_1_based = max(
                        1,
                        min(available_frames, int(loop_start_index)),
                    )

                    stop_idx_1_based = max(
                        1,
                        min(available_frames, int(loop_stop_index)),
                    )

                    if int(stop_idx_1_based) < int(start_idx_1_based):

                        raise ValueError(
                            f"Invalid loop range: start {int(start_idx_1_based)} > stop {int(stop_idx_1_based)}"
                        )

                    loop_start_0_based = int(start_idx_1_based - 1)

                    total_frames = int(stop_idx_1_based - start_idx_1_based + 1)

                hold_idx_1_based = (
                    0
                    if dynamic_phase_cal
                    else max(0, min(int(hold_pattern_index), int(total_frames)))
                )

                hold_idx_0_based = (
                    int(hold_idx_1_based - 1) if int(hold_idx_1_based) > 0 else -1
                )

                cached_periods = (-1, -1)

                cached_ramp = np.zeros((expected_h, expected_w), dtype=np.float32)

                cached_lens_params: Optional[
                    tuple[float, float, float, float]
                ] = None

                cached_lens = np.zeros((expected_h, expected_w), dtype=np.float32)

                def _compose_overlaid_phase(display_idx: int) -> np.ndarray:
                    nonlocal cached_periods, cached_ramp, cached_lens_params, cached_lens
                    if dynamic_phase_cal:
                        base_phase = self._build_phase_calibration_pattern_for_index(
                            display_idx, (expected_h, expected_w)
                        )
                        # For PSI acquisition, keep non-target super-pixels at background phase.
                        global_grating_phase = np.zeros(
                            (expected_h, expected_w), dtype=np.float32
                        )
                        zernike_phase = np.zeros(
                            (expected_h, expected_w), dtype=np.float32
                        )
                        focal_lens_phase = np.zeros(
                            (expected_h, expected_w), dtype=np.float32
                        )

                    else:

                        base_phase = np.mod(phase_patterns[display_idx], 2.0 * np.pi)
                        periods = (
                            int(self._grating_period_x_px),
                            int(self._grating_period_y_px),
                        )
                        if periods != cached_periods:
                            cached_ramp = self._build_grating_phase_ramp(
                                (expected_h, expected_w), periods[0], periods[1]
                            )
                            cached_periods = periods
                        global_grating_phase = cached_ramp
                        zernike_phase = self._build_zernike_phase_map(
                            (expected_h, expected_w)
                        )

                        focal_length_m = float(self._focal_lens_f_m)

                        if abs(float(focal_length_m)) > 1e-15:

                            lens_params = (
                                round(float(focal_length_m), 12),
                                round(float(self._plm_pixel_pitch_x_m), 12),
                                round(float(self._plm_pixel_pitch_y_m), 12),
                                round(float(self._focal_lens_wavelength_m), 12),
                            )

                            if lens_params != cached_lens_params:

                                cached_lens = self._build_focal_lens_phase_map(
                                    (expected_h, expected_w),
                                    focal_length_m=float(focal_length_m),
                                    pitch_x_m=float(self._plm_pixel_pitch_x_m),
                                    pitch_y_m=float(self._plm_pixel_pitch_y_m),
                                    wavelength_m=float(self._focal_lens_wavelength_m),
                                )

                                cached_lens_params = lens_params

                            focal_lens_phase = cached_lens

                        else:

                            focal_lens_phase = np.zeros(
                                (expected_h, expected_w), dtype=np.float32
                            )

                            cached_lens_params = None

                    phase_correction = self._get_active_phase_correction_map(
                        (expected_h, expected_w)
                    )

                    if phase_correction is None:
                        composed = np.mod(
                            base_phase
                            + global_grating_phase
                            + focal_lens_phase
                            + zernike_phase,
                            2.0 * np.pi,
                        )
                        return self._apply_hologram_position(
                            self._apply_hologram_scaling(composed)
                        )

                    composed = np.mod(
                        base_phase
                        + global_grating_phase
                        + focal_lens_phase
                        + zernike_phase
                        + phase_correction,
                        2.0 * np.pi,
                    )
                    return self._apply_hologram_position(
                        self._apply_hologram_scaling(composed)
                    )

                #def precompute_phase_lut_for_pattern_loop() -> np.ndarray:


                def loop_callback() -> None:
                    nonlocal frame_idx, last_update, cached_periods, cached_ramp
                    if self._plm_stop_event.is_set():
                        raise EventLoopExit
                    
                    now = perf_counter()

                    if now - last_update >= interval_s:

                        t1 = time()
                        display_idx = (
                            int(loop_start_0_based + hold_idx_0_based)
                            if int(hold_idx_0_based) >= 0
                            else int(loop_start_0_based + frame_idx)
                        )
                        overlaid_phase = _compose_overlaid_phase(display_idx)
                        
                        #self.plm_phase_preview_updated.emit(
                        #    np.asarray(overlaid_phase, dtype=np.float32)
                        #)

                        state_indices = np.digitize(
                            overlaid_phase, buckets, right=False
                        ).astype(np.int32)
                        t2 = time()
                        print(f"Phase composition and digitization timing: {t2 - t1:.3f}s")


                        t1 = time()
                        #'''
                        state_indices = np.clip(state_indices, 0, n_states - 1)
                        phase_for_lut = phase_values[state_indices]
                        bmp = plm.process_phase_map(phase_for_lut)
                        #'''
                        #bmp = np.zeros((2716, 1600), dtype=np.uint8)
                        bmp_image = Image.fromarray(bmp)
                        t2 = time()

                        win.clear()
                        win.load(bmp_image)
                        
                        t3 = time()
                        print(f"Loop timing: compose {t2 - t1:.3f}s, display {t3 - t2:.3f}s")

                        counter_idx_1_based = (
                            int(hold_idx_1_based)
                            if int(hold_idx_0_based) >= 0
                            else int(frame_idx + 1)
                        )

                        self.plm_counter_updated.emit(counter_idx_1_based, total_frames)

                        if int(hold_idx_0_based) < 0:

                            frame_idx = (frame_idx + 1) % total_frames
                        last_update = now

                win.loop_callback = loop_callback
                initial_display_idx = (
                    int(loop_start_0_based + hold_idx_0_based)
                    if int(hold_idx_0_based) >= 0
                    else int(loop_start_0_based)
                )
                initial_phase_overlaid = _compose_overlaid_phase(initial_display_idx)
                self.plm_phase_preview_updated.emit(
                    np.asarray(initial_phase_overlaid, dtype=np.float32)
                )

                initial_state_indices = np.digitize(
                    initial_phase_overlaid, buckets, right=False
                ).astype(np.int32)

                initial_state_indices = np.clip(initial_state_indices, 0, n_states - 1)
                initial_phase = phase_values[initial_state_indices]

                win.load(Image.fromarray(plm.process_phase_map(initial_phase)))
                initial_counter_idx_1_based = (
                    int(hold_idx_1_based) if int(hold_idx_0_based) >= 0 else 1
                )
                self.plm_counter_updated.emit(initial_counter_idx_1_based, total_frames)

                win.run()


            if self._plm_stop_event.is_set():
                message = "PLM loop stopped."

            else:
                message = "PLM loop exited."

        except Exception as exc:
            message = f"PLM loop error: {exc}"

        finally:
            self.plm_loop_finished.emit(message)