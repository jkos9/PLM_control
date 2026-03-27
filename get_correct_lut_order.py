from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _parse_timestamp(value: str) -> datetime:
	return datetime.fromisoformat(value)


def load_phase_measurements(csv_path: Path):
	rows = []
	with csv_path.open("r", newline="", encoding="utf-8") as handle:
		reader = csv.DictReader(handle)
		for row in reader:
			try:
				ts = _parse_timestamp(row["timestamp_iso"])
				dphi_wrapped = float(row["dphi_rad_wrapped"])
				dphi_0_2pi = float(row.get("dphi_rad_0_2pi", (dphi_wrapped + 2.0 * np.pi) % (2.0 * np.pi)))
				phi1 = float(row.get("phi1_rad", "nan"))
				phi2 = float(row.get("phi2_rad", "nan"))
				pattern_index = int(float(row["pattern_index"])) if row["pattern_index"] not in ("", "None") else -1
				samples_per_pattern = int(float(row.get("samples_per_pattern_max", "1")))
				sample_in_pattern = int(float(row.get("sample_in_pattern", "1")))
				rows.append(
					(
						ts,
						dphi_wrapped,
						dphi_0_2pi,
						phi1,
						phi2,
						pattern_index,
						samples_per_pattern,
						sample_in_pattern,
					)
				)
			except Exception:
				continue

	rows.sort(key=lambda item: item[0])
	return rows


def drop_first_datapoint_on_frame_switch(rows):
	if not rows:
		return rows

	filtered = []
	last_pattern_index = None
	for row in rows:
		pattern_index = row[5]
		sample_in_pattern = row[7]
		frame_switched = last_pattern_index is None or pattern_index != last_pattern_index

		if sample_in_pattern == 1 or frame_switched:
			last_pattern_index = pattern_index
			continue

		filtered.append(row)
		last_pattern_index = pattern_index

	return filtered


def compute_delta_phase_0_2pi(rows):
	dphi_0_2pi_logged = np.array([r[2] for r in rows], dtype=np.float64)
	phi1 = np.array([r[3] for r in rows], dtype=np.float64)
	phi2 = np.array([r[4] for r in rows], dtype=np.float64)
	phi_valid_mask = np.isfinite(phi1) & np.isfinite(phi2)

	if np.any(phi_valid_mask):
		dphi = np.full_like(phi1, np.nan)
		dphi[phi_valid_mask] = np.mod(phi1[phi_valid_mask] - phi2[phi_valid_mask], 2.0 * np.pi)
		phase_label = "Δφ from (phi1_rad - phi2_rad) in [0, 2π)"
	else:
		dphi = dphi_0_2pi_logged
		phase_label = "Δφ in [0, 2π)"

	return dphi, phase_label


def print_mean_delta_phase_per_frame(rows) -> None:
	if not rows:
		print("No rows available for per-frame averaging.")
		return

	dphi_plot, phase_label = compute_delta_phase_0_2pi(rows)
	pattern_index = np.array([r[5] for r in rows], dtype=np.int32)
	frame_ids = []
	mean_values = []

	print(f"Per-frame mean {phase_label}:")
	for idx in sorted(set(pattern_index.tolist())):
		mask = pattern_index == idx
		vals = dphi_plot[mask]
		vals = vals[np.isfinite(vals)]
		if vals.size == 0:
			print(f"  frame {idx}: n=0, mean=nan")
			continue
		mean_val = float(np.mean(vals))
		frame_ids.append(int(idx))
		mean_values.append(mean_val)
		print(f"  frame {idx}: n={vals.size}, mean={mean_val:.6f} rad")

	mean_array = np.array(mean_values, dtype=np.float64)
	print("\nCopy/paste arrays:")
	print(f"frame_ids = {frame_ids}")
	print(f"mean_delta_phase_per_frame = np.array({mean_array.tolist()}, dtype=np.float64)")


def plot_phase_chronological(rows):
	if not rows:
		raise RuntimeError("No valid rows found in CSV.")

	timestamps = np.array([r[0] for r in rows])
	phi1 = np.array([r[3] for r in rows], dtype=np.float64)
	phi2 = np.array([r[4] for r in rows], dtype=np.float64)
	pattern_index = np.array([r[5] for r in rows], dtype=np.int32)
	sample_in_pattern = np.array([r[7] for r in rows], dtype=np.int32)
	dphi_plot, phase_label = compute_delta_phase_0_2pi(rows)

	fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)

	point_phase_ax = axes[0]
	point_phase_ax.plot(timestamps, phi1, color="tab:green", linewidth=1.2, label="phi1_rad (P1)")
	point_phase_ax.plot(timestamps, phi2, color="tab:purple", linewidth=1.2, label="phi2_rad (P2)")
	point_phase_ax.set_ylabel("Point phase (rad)")
	point_phase_ax.set_title("Point phases over time")
	#point_phase_ax.set_ylim(-np.pi, np.pi)
	point_phase_ax.grid(True, alpha=0.3)
	point_phase_ax.legend(loc="upper right")

	phase_ax = axes[1]
	phase_ax.plot(timestamps, dphi_plot, color="tab:blue", linewidth=1.2, label=phase_label)
	phase_ax.scatter(
		timestamps,
		dphi_plot,
		c=pattern_index,
		cmap="viridis",
		s=20,
		alpha=0.9,
		label="Samples",
	)
	phase_ax.set_ylabel("Δφ (rad)")
	phase_ax.set_title("Relative phase in [0, 2π) over time")
	#phase_ax.set_ylim(0.0, 2.0 * np.pi)
	phase_ax.grid(True, alpha=0.3)
	phase_ax.legend(loc="upper right")

	pattern_ax = axes[2]
	pattern_ax.step(timestamps, pattern_index, where="post", color="tab:orange", linewidth=1.2)
	pattern_ax.scatter(
		timestamps,
		pattern_index,
		c=sample_in_pattern,
		cmap="plasma",
		s=18,
		alpha=0.9,
	)
	pattern_ax.set_ylabel("Pattern index")
	pattern_ax.set_xlabel("Timestamp")
	pattern_ax.set_title("Pattern index displayed at sample time")
	pattern_ax.grid(True, alpha=0.3)

	plt.show()


def main() -> None:
	csv_path = Path("logs") / "phase_measurements_11.csv"
	if not csv_path.exists():
		raise FileNotFoundError(f"Could not find log file: {csv_path}")

	rows = load_phase_measurements(csv_path)
	rows = drop_first_datapoint_on_frame_switch(rows)
	if not rows:
		raise RuntimeError("No valid rows left after dropping first sample at each frame switch.")
	print_mean_delta_phase_per_frame(rows)
	plot_phase_chronological(rows)



if __name__ == "__main__":
	main()



