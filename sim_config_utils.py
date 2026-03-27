import torch


def _finalize_config(config):
    max_mg = int(config["max_mg"])
    config["mode_count"] = max_mg * (max_mg + 1) // 2
    return config


def get_sim_config(overrides=None):
    config = {
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        "max_mg": 5,
        "lambda_": 1550e-9,
        "downsample": 1,
        "plane_spacing": 0.15,
        "array_dist_to_first_plane": 5e-3,
        "plane_count": 8,
        "mf_din": 700e-6,
        "mf_dout": 700e-6,
        "k_space_filter": 0.1,
        "symmetric_masks": False,
        "iteration_count": 500,
        "boost": 1,
        "apply_smoothing": True,
        "smoothing_kernel_size": 3,
        "smoothing_sigma": 0.65,
        "learning_rate": 3e-1,
        "prop_tilt_x_deg": 0.0,
        "prop_tilt_y_deg": 0.0,
        "prop_recenter": True,
        "pixel_size_x": 16.2e-6,
        "pixel_size_y": 10.8e-6,
        "nx": 904,
        "ny": 800,
        "mask_offset": 1e-9,
    }

    if overrides:
        config.update(overrides)

    return _finalize_config(config)


def get_spatial_extent(config, unit_scale=1.0):
    nx = int(config["nx"])
    ny = int(config["ny"])
    dx = float(config["pixel_size_x"])
    dy = float(config["pixel_size_y"])

    half_width = 0.5 * ny * dx * unit_scale
    half_height = 0.5 * nx * dy * unit_scale
    return (-half_width, half_width, -half_height, half_height)
