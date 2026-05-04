"""
Utilities for ephys analysis - helper functions and plotting functions
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


# ==========================================================
# === HELPER FUNCTIONS ====================================
# ==========================================================

def flatten_array_of_arrays(arr):
    out = []
    for x in arr:
        if isinstance(x, (list, np.ndarray)):
            out.extend(list(x))
        elif pd.notna(x):
            out.append(float(x))
    return np.array(out, dtype=float)


def extract_y_from_locations(locations, name="locations"):
    if isinstance(locations, dict):
        if "y" in locations:
            return np.asarray(locations["y"], dtype=float).reshape(-1)

        for key in ["sampled_locations_um", "unit_locations", "locations"]:
            if key in locations:
                arr = np.asarray(locations[key])

                if arr.dtype.names is not None:
                    if "y" in arr.dtype.names:
                        return np.asarray(arr["y"], dtype=float).reshape(-1)
                    raise ValueError(f"{name}: structured array has no 'y' field. Fields: {arr.dtype.names}")

                if arr.ndim == 2 and arr.shape[1] >= 2:
                    return np.asarray(arr[:, 1], dtype=float).reshape(-1)

                if arr.ndim == 1:
                    return np.asarray(arr, dtype=float).reshape(-1)

                raise ValueError(f"{name}: unsupported array shape in key '{key}': {arr.shape}")

        raise ValueError(f"{name}: unknown dict keys: {list(locations.keys())}")

    arr = np.asarray(locations)

    if arr.dtype.names is not None:
        if "y" in arr.dtype.names:
            return np.asarray(arr["y"], dtype=float).reshape(-1)
        raise ValueError(f"{name}: structured array has no 'y' field. Fields: {arr.dtype.names}")

    if arr.ndim == 2 and arr.shape[1] >= 2:
        return np.asarray(arr[:, 1], dtype=float).reshape(-1)

    if arr.ndim == 1:
        return np.asarray(arr, dtype=float).reshape(-1)

    raise ValueError(f"{name}: unsupported shape {arr.shape}")


def gaussian_kernel1d(sigma, radius=None):
    if sigma <= 0:
        return np.array([1.0], dtype=float)
    if radius is None:
        radius = max(1, int(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    k /= k.sum()
    return k


def gaussian_smooth_1d(arr, sigma, axis):
    if sigma <= 0:
        return arr
    kernel = gaussian_kernel1d(sigma)
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=axis, arr=arr)


def gaussian_smooth_reflect_1d(arr, sigma):
    if sigma <= 0:
        return arr.copy()
    kernel = gaussian_kernel1d(sigma)
    pad = len(kernel) // 2
    padded = np.pad(arr, pad_width=pad, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def zscore_from_baseline(curve, time_centers, baseline_start, baseline_end):
    baseline_mask = (time_centers >= baseline_start) & (time_centers < baseline_end)
    if baseline_mask.sum() == 0:
        return np.full_like(curve, np.nan, dtype=float)
    baseline = curve[baseline_mask]
    mu = np.mean(baseline)
    sigma = np.std(baseline)
    if sigma < 1e-12:
        return curve - mu
    return (curve - mu) / sigma


def zscore_from_baseline_mask(curve, baseline_mask):
    baseline = curve[baseline_mask]
    mu = np.mean(baseline)
    sigma = np.std(baseline)
    if sigma < 1e-12:
        return curve - mu
    return (curve - mu) / sigma


def histogram_rate(times, bins):
    counts, _ = np.histogram(times, bins=bins)
    bin_width = np.diff(bins)[0]
    return counts / bin_width


def build_profiles_for_band(
    spike_times_beh_band,
    valid_bout_starts,
    valid_first_licks,
    valid_last_licks,
    pre_window_s,
    post_window_s,
    time_bin_s,
    n_bins_start_to_first,
    n_bins_first_to_last,
):
    pre_bins = np.arange(-pre_window_s, 0 + time_bin_s, time_bin_s)
    post_bins = np.arange(0, post_window_s + time_bin_s, time_bin_s)

    n_pre = len(pre_bins) - 1
    n_mid1 = n_bins_start_to_first
    n_mid2 = n_bins_first_to_last
    n_post = len(post_bins) - 1

    baseline_mask = np.r_[
        np.ones(n_pre, dtype=bool),
        np.zeros(n_mid1 + n_mid2 + n_post, dtype=bool)
    ]

    all_profiles_z = []

    for bout_start, first_lick, last_lick in zip(valid_bout_starts, valid_first_licks, valid_last_licks):
        pre_start = bout_start - pre_window_s
        pre_stop = bout_start
        pre_mask = (spike_times_beh_band >= pre_start) & (spike_times_beh_band < pre_stop)
        pre_times = spike_times_beh_band[pre_mask] - bout_start
        pre_rate = histogram_rate(pre_times, pre_bins)

        mid1_duration = first_lick - bout_start
        if mid1_duration <= 0:
            continue

        mid1_mask = (spike_times_beh_band >= bout_start) & (spike_times_beh_band < first_lick)
        mid1_times_norm = (spike_times_beh_band[mid1_mask] - bout_start) / mid1_duration
        mid1_bins = np.linspace(0, 1, n_mid1 + 1)
        mid1_rate = np.histogram(mid1_times_norm, bins=mid1_bins)[0] / (mid1_duration / n_mid1)

        mid2_duration = last_lick - first_lick
        if mid2_duration <= 0:
            continue

        mid2_mask = (spike_times_beh_band >= first_lick) & (spike_times_beh_band < last_lick)
        mid2_times_norm = (spike_times_beh_band[mid2_mask] - first_lick) / mid2_duration
        mid2_bins = np.linspace(0, 1, n_mid2 + 1)
        mid2_rate = np.histogram(mid2_times_norm, bins=mid2_bins)[0] / (mid2_duration / n_mid2)

        post_start = last_lick
        post_stop = last_lick + post_window_s
        post_mask = (spike_times_beh_band >= post_start) & (spike_times_beh_band < post_stop)
        post_times = spike_times_beh_band[post_mask] - last_lick
        post_rate = histogram_rate(post_times, post_bins)

        profile = np.concatenate([pre_rate, mid1_rate, mid2_rate, post_rate])
        profile_z = zscore_from_baseline_mask(profile, baseline_mask)
        all_profiles_z.append(profile_z)

    if len(all_profiles_z) == 0:
        raise ValueError("No bout profiles could be built for this depth band.")

    all_profiles_z = np.vstack(all_profiles_z)
    return (
        all_profiles_z,
        np.mean(all_profiles_z, axis=0),
        np.std(all_profiles_z, axis=0),
    )


def hex_to_rgb01(hex_color):
    hex_color = hex_color.lstrip("#")
    return np.array([
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16)
    ], dtype=float) / 255.0


def smooth_positive(x, sigma):
    y = gaussian_smooth_1d(x, sigma=sigma, axis=0)
    return np.maximum(y, 0)


# ==========================================================
# === PLOTTING FUNCTIONS ==================================
# ==========================================================

def plot_heatmap_probe(
    heatmap,
    time_bins,
    depth_bins,
    spike_density_smooth,
    depth_centers,
    bout_times=None,
    bout_color="#4FC3F7",
    title="Spike activity along probe",
):
    """Plot heatmap of spike activity with optional bout markers"""
    vmin = np.percentile(heatmap, 10)
    vmax = np.percentile(heatmap, 99)

    if vmax <= vmin:
        vmax = vmin + 1e-3

    fig, (ax0, ax1) = plt.subplots(
        1, 2,
        figsize=(13, 9),
        gridspec_kw={"width_ratios": [4, 1]},
        sharey=True
    )

    im = ax0.imshow(
        heatmap,
        aspect="auto",
        origin="lower",
        extent=[time_bins[0], time_bins[-1], depth_bins[0], depth_bins[-1]],
        cmap="magma",
        vmin=vmin,
        vmax=vmax
    )

    if bout_times is not None:
        for t in bout_times:
            ax0.axvline(t, color=bout_color, linewidth=0.9, alpha=0.85)

    ax0.set_title(title)
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("Depth (µm)")

    cbar = plt.colorbar(im, ax=ax0)
    cbar.set_label("Spike rate")

    ax1.plot(spike_density_smooth, depth_centers, linewidth=2, color="black")
    ax1.fill_betweenx(depth_centers, 0, spike_density_smooth, alpha=0.3, color="gray")
    ax1.set_title("Spike density")
    ax1.set_xlabel("Spike count")
    ax1.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_random_bouts_depth_bands(
    spike_times_beh_filtered, spike_depths_filtered, bout_starts, lick_rewarded, lick_nonrewarded,
    valid_bout_starts, DEPTH_BAND_1, DEPTH_BAND_2, DEPTH_BAND_3,
    BAND_NAME_1, BAND_NAME_2, BAND_NAME_3,
    COLOR_BAND_1, COLOR_BAND_2, COLOR_BAND_3,
    REWARDED_COLOR, NONREWARDED_COLOR,
    N_BOUTS_TO_PLOT, RANDOM_SEED,
    RAMBOUTS_T_BEFORE, RAMBOUTS_T_AFTER, RAMBOUTS_TIME_BIN_S, RAMBOUTS_SMOOTH_SIGMA_CURVE
):
    """Plot random valid bouts with activity by depth bands"""
    # Select random valid bouts
    if len(valid_bout_starts) == 0:
        raise ValueError("No valid bouts found.")

    rng = np.random.default_rng(RANDOM_SEED)
    n_keep = min(N_BOUTS_TO_PLOT, len(valid_bout_starts))
    selected_bouts = np.sort(rng.choice(valid_bout_starts, size=n_keep, replace=False))

    # Compute global z-score stats per depth band
    global_t_start = np.floor(np.min(spike_times_beh_filtered))
    global_t_stop = np.ceil(np.max(spike_times_beh_filtered)) + RAMBOUTS_TIME_BIN_S
    global_time_bins = np.arange(global_t_start, global_t_stop + RAMBOUTS_TIME_BIN_S, RAMBOUTS_TIME_BIN_S)

    mask_band1 = (spike_depths_filtered >= DEPTH_BAND_1[0]) & (spike_depths_filtered < DEPTH_BAND_1[1])
    mask_band2 = (spike_depths_filtered >= DEPTH_BAND_2[0]) & (spike_depths_filtered < DEPTH_BAND_2[1])
    mask_band3 = spike_depths_filtered >= DEPTH_BAND_3[0]

    global_counts1, _ = np.histogram(spike_times_beh_filtered[mask_band1], bins=global_time_bins)
    global_counts2, _ = np.histogram(spike_times_beh_filtered[mask_band2], bins=global_time_bins)
    global_counts3, _ = np.histogram(spike_times_beh_filtered[mask_band3], bins=global_time_bins)

    global_rate1 = global_counts1 / RAMBOUTS_TIME_BIN_S
    global_rate2 = global_counts2 / RAMBOUTS_TIME_BIN_S
    global_rate3 = global_counts3 / RAMBOUTS_TIME_BIN_S

    mu1, sigma1 = np.mean(global_rate1), np.std(global_rate1)
    mu2, sigma2 = np.mean(global_rate2), np.std(global_rate2)
    mu3, sigma3 = np.mean(global_rate3), np.std(global_rate3)

    sigma1 = max(sigma1, 1e-12)
    sigma2 = max(sigma2, 1e-12)
    sigma3 = max(sigma3, 1e-12)

    print("Global z-score stats (depth bands):")
    print(f"  {BAND_NAME_1}: mu={mu1:.3f}, sigma={sigma1:.3f}")
    print(f"  {BAND_NAME_2}: mu={mu2:.3f}, sigma={sigma2:.3f}")
    print(f"  {BAND_NAME_3}: mu={mu3:.3f}, sigma={sigma3:.3f}")

    # Plot
    ncols = 1
    nrows = len(selected_bouts)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.2 * nrows), sharex=False, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for i, bout_t in enumerate(selected_bouts):
        ax = axes[i]
        bout_idx = np.where(bout_starts == bout_t)[0]
        
        if len(bout_idx) == 0:
            ax.set_title(f"Bout {i+1} | impossible to find index")
            ax.grid(alpha=0.3)
            continue

        bout_idx = bout_idx[0]
        bout_start = bout_starts[bout_idx]
        bout_stop = bout_starts[bout_idx + 1] if bout_idx < len(bout_starts) - 1 else np.inf

        bout_rewarded = lick_rewarded[(lick_rewarded >= bout_start) & (lick_rewarded < bout_stop)]
        bout_nonrewarded = lick_nonrewarded[(lick_nonrewarded >= bout_start) & (lick_nonrewarded < bout_stop)]
        bout_licks = np.sort(np.concatenate([bout_rewarded, bout_nonrewarded]))

        if len(bout_licks) == 0:
            ax.set_title(f"Bout {i+1} | no licks")
            ax.grid(alpha=0.3)
            continue

        first_lick = bout_licks[0]
        last_lick = bout_licks[-1]

        win_start = first_lick - RAMBOUTS_T_BEFORE
        win_stop = last_lick + RAMBOUTS_T_AFTER

        rel_time_bins = np.arange(-RAMBOUTS_T_BEFORE, (last_lick - first_lick) + RAMBOUTS_T_AFTER + RAMBOUTS_TIME_BIN_S, RAMBOUTS_TIME_BIN_S)
        time_centers = 0.5 * (rel_time_bins[:-1] + rel_time_bins[1:])

        spike_mask = (spike_times_beh_filtered >= win_start) & (spike_times_beh_filtered <= win_stop)
        local_spike_times = spike_times_beh_filtered[spike_mask] - first_lick
        local_spike_depths = spike_depths_filtered[spike_mask]

        local_mask_band1 = (local_spike_depths >= DEPTH_BAND_1[0]) & (local_spike_depths < DEPTH_BAND_1[1])
        local_mask_band2 = (local_spike_depths >= DEPTH_BAND_2[0]) & (local_spike_depths < DEPTH_BAND_2[1])
        local_mask_band3 = local_spike_depths >= DEPTH_BAND_3[0]

        counts1, _ = np.histogram(local_spike_times[local_mask_band1], bins=rel_time_bins)
        counts2, _ = np.histogram(local_spike_times[local_mask_band2], bins=rel_time_bins)
        counts3, _ = np.histogram(local_spike_times[local_mask_band3], bins=rel_time_bins)

        rate1 = counts1 / RAMBOUTS_TIME_BIN_S
        rate2 = counts2 / RAMBOUTS_TIME_BIN_S
        rate3 = counts3 / RAMBOUTS_TIME_BIN_S

        rate1_z = (rate1 - mu1) / sigma1
        rate2_z = (rate2 - mu2) / sigma2
        rate3_z = (rate3 - mu3) / sigma3

        rate1_s = gaussian_smooth_1d(rate1_z, RAMBOUTS_SMOOTH_SIGMA_CURVE, axis=0)
        rate2_s = gaussian_smooth_1d(rate2_z, RAMBOUTS_SMOOTH_SIGMA_CURVE, axis=0)
        rate3_s = gaussian_smooth_1d(rate3_z, RAMBOUTS_SMOOTH_SIGMA_CURVE, axis=0)

        local_rewarded = lick_rewarded[(lick_rewarded >= win_start) & (lick_rewarded <= win_stop)] - first_lick
        local_nonrewarded = lick_nonrewarded[(lick_nonrewarded >= win_start) & (lick_nonrewarded <= win_stop)] - first_lick

        ax.plot(time_centers, rate1_s, color=COLOR_BAND_1, linewidth=2.0)
        ax.plot(time_centers, rate2_s, color=COLOR_BAND_2, linewidth=2.0)
        ax.plot(time_centers, rate3_s, color=COLOR_BAND_3, linewidth=2.2)

        for t in local_rewarded:
            ax.axvline(t, color=REWARDED_COLOR, linewidth=1.2, alpha=0.30)
        for t in local_nonrewarded:
            ax.axvline(t, color=NONREWARDED_COLOR, linewidth=1.2, alpha=0.30)

        ax.set_title(f"Bout {i+1} | 1st lick = {first_lick:.2f}s | last lick = {last_lick:.2f}s", fontsize=12)
        ax.grid(alpha=0.3)

    for ax in axes:
        ax.set_ylabel("Spike density\n(global z-score)")

    axes[-1].set_xlabel("Time from first lick of bout (s)")

    handles = [
        plt.Line2D([0], [0], color=COLOR_BAND_1, lw=2.0, label=BAND_NAME_1),
        plt.Line2D([0], [0], color=COLOR_BAND_2, lw=2.0, label=BAND_NAME_2),
        plt.Line2D([0], [0], color=COLOR_BAND_3, lw=2.2, label=BAND_NAME_3),
        plt.Line2D([0], [0], color=REWARDED_COLOR, lw=2.0, label="Rewarded licks"),
        plt.Line2D([0], [0], color=NONREWARDED_COLOR, lw=2.0, label="Non-rewarded licks"),
    ]

    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=5, frameon=False, fontsize=10.5)
    plt.suptitle("Spike density by depth band", y=1.02, fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()


def plot_random_bouts_rgb_mixing(
    spike_times_beh_filtered, spike_depths_filtered, bout_starts, lick_rewarded, lick_nonrewarded,
    valid_bout_starts, DEPTH_BAND_1, DEPTH_BAND_2, DEPTH_BAND_3,
    BAND_NAME_1, BAND_NAME_2, BAND_NAME_3,
    COLOR_BAND_1, COLOR_BAND_2, COLOR_BAND_3,
    REWARDED_COLOR, NONREWARDED_COLOR, BOUT_COLOR, GLOBAL_CURVE_COLOR,
    N_BOUTS_TO_PLOT, RANDOM_SEED,
    RGBMIX_T_BEFORE, RGBMIX_T_AFTER, RGBMIX_TIME_BIN_S, RGBMIX_SMOOTH_SIGMA_CURVE
):
    """Plot random valid bouts with RGB mixing of depth bands"""
    # Select random valid bouts
    rng = np.random.default_rng(RANDOM_SEED)
    n_keep = min(N_BOUTS_TO_PLOT, len(valid_bout_starts))
    selected_bouts = np.sort(rng.choice(valid_bout_starts, size=n_keep, replace=False))

    # Compute global z-score stats for RGB mixing
    global_t_start = np.floor(np.min(spike_times_beh_filtered))
    global_t_stop = np.ceil(np.max(spike_times_beh_filtered)) + RGBMIX_TIME_BIN_S
    global_time_bins = np.arange(global_t_start, global_t_stop + RGBMIX_TIME_BIN_S, RGBMIX_TIME_BIN_S)

    mask_band1 = (spike_depths_filtered >= DEPTH_BAND_1[0]) & (spike_depths_filtered < DEPTH_BAND_1[1])
    mask_band2 = (spike_depths_filtered >= DEPTH_BAND_2[0]) & (spike_depths_filtered < DEPTH_BAND_2[1])
    mask_band3 = spike_depths_filtered >= DEPTH_BAND_3[0]

    global_counts1, _ = np.histogram(spike_times_beh_filtered[mask_band1], bins=global_time_bins)
    global_counts2, _ = np.histogram(spike_times_beh_filtered[mask_band2], bins=global_time_bins)
    global_counts3, _ = np.histogram(spike_times_beh_filtered[mask_band3], bins=global_time_bins)

    global_rate1 = global_counts1 / RGBMIX_TIME_BIN_S
    global_rate2 = global_counts2 / RGBMIX_TIME_BIN_S
    global_rate3 = global_counts3 / RGBMIX_TIME_BIN_S

    mu1, sigma1 = np.mean(global_rate1), np.std(global_rate1)
    mu2, sigma2 = np.mean(global_rate2), np.std(global_rate2)
    mu3, sigma3 = np.mean(global_rate3), np.std(global_rate3)

    sigma1 = max(sigma1, 1e-12)
    sigma2 = max(sigma2, 1e-12)
    sigma3 = max(sigma3, 1e-12)

    c1 = hex_to_rgb01(COLOR_BAND_1)
    c2 = hex_to_rgb01(COLOR_BAND_2)
    c3 = hex_to_rgb01(COLOR_BAND_3)

    print("Global z-score stats (RGB mixing):")
    print(f"  {BAND_NAME_1}: mu={mu1:.3f}, sigma={sigma1:.3f}")
    print(f"  {BAND_NAME_2}: mu={mu2:.3f}, sigma={sigma2:.3f}")
    print(f"  {BAND_NAME_3}: mu={mu3:.3f}, sigma={sigma3:.3f}")

    # Plot
    ncols = 1
    nrows = len(selected_bouts)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.4 * nrows), sharex=False, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for i, bout_t in enumerate(selected_bouts):
        ax = axes[i]

        win_start = bout_t - RGBMIX_T_BEFORE
        win_stop = bout_t + RGBMIX_T_AFTER

        rel_time_bins = np.arange(-RGBMIX_T_BEFORE, RGBMIX_T_AFTER + RGBMIX_TIME_BIN_S, RGBMIX_TIME_BIN_S)
        time_centers = 0.5 * (rel_time_bins[:-1] + rel_time_bins[1:])

        spike_mask = (spike_times_beh_filtered >= win_start) & (spike_times_beh_filtered <= win_stop)
        local_spike_times = spike_times_beh_filtered[spike_mask] - bout_t
        local_spike_depths = spike_depths_filtered[spike_mask]

        local_mask_band1 = (local_spike_depths >= DEPTH_BAND_1[0]) & (local_spike_depths < DEPTH_BAND_1[1])
        local_mask_band2 = (local_spike_depths >= DEPTH_BAND_2[0]) & (local_spike_depths < DEPTH_BAND_2[1])
        local_mask_band3 = local_spike_depths >= DEPTH_BAND_3[0]

        counts1, _ = np.histogram(local_spike_times[local_mask_band1], bins=rel_time_bins)
        counts2, _ = np.histogram(local_spike_times[local_mask_band2], bins=rel_time_bins)
        counts3, _ = np.histogram(local_spike_times[local_mask_band3], bins=rel_time_bins)

        rate1 = counts1 / RGBMIX_TIME_BIN_S
        rate2 = counts2 / RGBMIX_TIME_BIN_S
        rate3 = counts3 / RGBMIX_TIME_BIN_S

        rate1_z = (rate1 - mu1) / sigma1
        rate2_z = (rate2 - mu2) / sigma2
        rate3_z = (rate3 - mu3) / sigma3

        rate1_s = gaussian_smooth_1d(rate1_z, RGBMIX_SMOOTH_SIGMA_CURVE, axis=0)
        rate2_s = gaussian_smooth_1d(rate2_z, RGBMIX_SMOOTH_SIGMA_CURVE, axis=0)
        rate3_s = gaussian_smooth_1d(rate3_z, RGBMIX_SMOOTH_SIGMA_CURVE, axis=0)

        global_curve = (rate1_s + rate2_s + rate3_s) / 3.0

        p1 = smooth_positive(rate1_s, sigma=0.8)
        p2 = smooth_positive(rate2_s, sigma=0.8)
        p3 = smooth_positive(rate3_s, sigma=0.8)

        weight_sum = p1 + p2 + p3
        valid_color_mask = weight_sum > 1e-12

        w1 = np.zeros_like(weight_sum)
        w2 = np.zeros_like(weight_sum)
        w3 = np.zeros_like(weight_sum)

        w1[valid_color_mask] = p1[valid_color_mask] / weight_sum[valid_color_mask]
        w2[valid_color_mask] = p2[valid_color_mask] / weight_sum[valid_color_mask]
        w3[valid_color_mask] = p3[valid_color_mask] / weight_sum[valid_color_mask]

        rgb = np.zeros((len(time_centers), 3), dtype=float)
        rgb[:] = 0.85
        rgb[valid_color_mask] = (
            w1[valid_color_mask, None] * c1
            + w2[valid_color_mask, None] * c2
            + w3[valid_color_mask, None] * c3
        )

        rgb = 0.15 + 0.85 * rgb
        rgb = np.clip(rgb, 0, 1)

        local_rewarded = lick_rewarded[(lick_rewarded >= win_start) & (lick_rewarded <= win_stop)] - bout_t
        local_nonrewarded = lick_nonrewarded[(lick_nonrewarded >= win_start) & (lick_nonrewarded <= win_stop)] - bout_t

        for j in range(len(time_centers)):
            ax.fill_between(
                [rel_time_bins[j], rel_time_bins[j + 1]],
                [0, 0],
                [global_curve[j], global_curve[j]],
                color=rgb[j],
                alpha=0.65,
                linewidth=0
            )

        ax.plot(time_centers, global_curve, color=GLOBAL_CURVE_COLOR, linewidth=2.6, zorder=3)
        ax.axvline(0, color=BOUT_COLOR, linewidth=2.2, alpha=0.95)

        for t in local_rewarded:
            ax.axvline(t, color=REWARDED_COLOR, linewidth=1.1, alpha=0.22)
        for t in local_nonrewarded:
            ax.axvline(t, color=NONREWARDED_COLOR, linewidth=1.1, alpha=0.22)

        ax.axhline(0, color="black", linewidth=0.8, alpha=0.45)
        ax.set_title(f"Bout {i+1} | center = {bout_t:.2f}s", fontsize=12)
        ax.grid(alpha=0.25)

    for ax in axes:
        ax.set_ylabel("Global activity\n(global z-score)")

    axes[-1].set_xlabel("Time from bout start (s)")

    handles = [
        plt.Line2D([0], [0], color=GLOBAL_CURVE_COLOR, lw=2.6, label="Global activity"),
        plt.Line2D([0], [0], color=COLOR_BAND_1, lw=6, alpha=0.7, label=BAND_NAME_1),
        plt.Line2D([0], [0], color=COLOR_BAND_2, lw=6, alpha=0.7, label=BAND_NAME_2),
        plt.Line2D([0], [0], color=COLOR_BAND_3, lw=6, alpha=0.7, label=BAND_NAME_3),
        plt.Line2D([0], [0], color=BOUT_COLOR, lw=2.0, label="Bout start"),
        plt.Line2D([0], [0], color=REWARDED_COLOR, lw=2.0, label="Rewarded licks"),
        plt.Line2D([0], [0], color=NONREWARDED_COLOR, lw=2.0, label="Non-rewarded licks"),
    ]

    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=7, frameon=False, fontsize=10.5)
    plt.suptitle("Global spike activity with RGB depth mixing", y=1.02, fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()


def plot_warped_profiles(
    spike_times_beh_filtered, spike_depths_filtered,
    valid_bout_starts, valid_first_licks, valid_last_licks,
    DEPTH_BAND_1, DEPTH_BAND_2, DEPTH_BAND_3,
    BAND_NAME_1, BAND_NAME_2, BAND_NAME_3,
    COLOR_BAND_1, COLOR_BAND_2, COLOR_BAND_3, STD_COLOR_BAND_3,
    START_COLOR, FIRSTLICK_COLOR, ENDLICK_COLOR,
    BG_PRE, BG_MID1, BG_MID2, BG_POST,
    WARPED_PRE_WINDOW_S, WARPED_POST_WINDOW_S, WARPED_TIME_BIN_S,
    WARPED_SMOOTH_SIGMA, WARPED_N_BINS_START_TO_FIRST, WARPED_N_BINS_FIRST_TO_LAST
):
    """Plot warped profiles by depth band"""
    # Split spikes by depth band
    mask_band1 = (spike_depths_filtered >= DEPTH_BAND_1[0]) & (spike_depths_filtered < DEPTH_BAND_1[1])
    mask_band2 = (spike_depths_filtered >= DEPTH_BAND_2[0]) & (spike_depths_filtered < DEPTH_BAND_2[1])
    mask_band3 = spike_depths_filtered >= DEPTH_BAND_3[0]

    spike_times_1 = spike_times_beh_filtered[mask_band1]
    spike_times_2 = spike_times_beh_filtered[mask_band2]
    spike_times_3 = spike_times_beh_filtered[mask_band3]

    print("Building warped profiles...")

    # Build warped profiles for each band
    profiles_1, mean_1, std_1 = build_profiles_for_band(
        spike_times_1, valid_bout_starts, valid_first_licks, valid_last_licks,
        WARPED_PRE_WINDOW_S, WARPED_POST_WINDOW_S, WARPED_TIME_BIN_S,
        WARPED_N_BINS_START_TO_FIRST, WARPED_N_BINS_FIRST_TO_LAST,
    )

    profiles_2, mean_2, std_2 = build_profiles_for_band(
        spike_times_2, valid_bout_starts, valid_first_licks, valid_last_licks,
        WARPED_PRE_WINDOW_S, WARPED_POST_WINDOW_S, WARPED_TIME_BIN_S,
        WARPED_N_BINS_START_TO_FIRST, WARPED_N_BINS_FIRST_TO_LAST,
    )

    profiles_3, mean_3, std_3 = build_profiles_for_band(
        spike_times_3, valid_bout_starts, valid_first_licks, valid_last_licks,
        WARPED_PRE_WINDOW_S, WARPED_POST_WINDOW_S, WARPED_TIME_BIN_S,
        WARPED_N_BINS_START_TO_FIRST, WARPED_N_BINS_FIRST_TO_LAST,
    )

    mean_1_s = gaussian_smooth_reflect_1d(mean_1, WARPED_SMOOTH_SIGMA)
    mean_2_s = gaussian_smooth_reflect_1d(mean_2, WARPED_SMOOTH_SIGMA)
    mean_3_s = gaussian_smooth_reflect_1d(mean_3, WARPED_SMOOTH_SIGMA)
    std_3_s = gaussian_smooth_reflect_1d(std_3, WARPED_SMOOTH_SIGMA)

    print(f"Bouts used: {profiles_1.shape[0]} (band1), {profiles_2.shape[0]} (band2), {profiles_3.shape[0]} (band3)")

    # Build x-axis with visual zones
    n_pre = int(WARPED_PRE_WINDOW_S / WARPED_TIME_BIN_S)
    n_mid1 = WARPED_N_BINS_START_TO_FIRST
    n_mid2 = WARPED_N_BINS_FIRST_TO_LAST
    n_post = int(WARPED_POST_WINDOW_S / WARPED_TIME_BIN_S)

    mid1_width = 1.0 / 3.0
    mid2_start = 1.0 + mid1_width
    post_start_x = mid2_start + 1.0
    x_end = post_start_x + 1.0

    x_pre = np.linspace(0.0, 1.0, n_pre, endpoint=False) + 0.5 / n_pre
    x_mid1 = np.linspace(1.0, 1.0 + mid1_width, n_mid1, endpoint=False) + 0.5 * (mid1_width / n_mid1)
    x_mid2 = np.linspace(mid2_start, post_start_x, n_mid2, endpoint=False) + 0.5 * (1.0 / n_mid2)
    x_post = np.linspace(post_start_x, x_end, n_post, endpoint=False) + 0.5 * (1.0 / n_post)

    x_all = np.concatenate([x_pre, x_mid1, x_mid2, x_post])

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6.2))

    ax.axvspan(0.0, 1.0, color=BG_PRE, alpha=1.0, zorder=0)
    ax.axvspan(1.0, 1.0 + mid1_width, color=BG_MID1, alpha=1.0, zorder=0)
    ax.axvspan(mid2_start, post_start_x, color=BG_MID2, alpha=1.0, zorder=0)
    ax.axvspan(post_start_x, x_end, color=BG_POST, alpha=1.0, zorder=0)

    ax.fill_between(x_all, mean_3_s - std_3_s, mean_3_s + std_3_s,
                    color=STD_COLOR_BAND_3, alpha=0.28, linewidth=0,
                    label=f"{BAND_NAME_3} ± 1 SD", zorder=1)

    ax.plot(x_all, mean_1_s, color=COLOR_BAND_1, linewidth=2.2, label=BAND_NAME_1, zorder=3)
    ax.plot(x_all, mean_2_s, color=COLOR_BAND_2, linewidth=2.2, label=BAND_NAME_2, zorder=3)
    ax.plot(x_all, mean_3_s, color=COLOR_BAND_3, linewidth=2.8, label=BAND_NAME_3, zorder=4)

    ax.axvline(1.0, color=START_COLOR, linewidth=2.2, alpha=0.95, label="Bout start")
    ax.axvline(1.0 + mid1_width, color=FIRSTLICK_COLOR, linewidth=2.2, alpha=0.95, label="First lick")
    ax.axvline(post_start_x, color=ENDLICK_COLOR, linewidth=2.2, alpha=0.95, label="Last lick")

    ax.set_xlim(0.0, x_end)
    ax.set_xticks(
        [0.0, 0.5, 1.0, 1.0 + mid1_width / 2, 1.0 + mid1_width, (mid2_start + post_start_x) / 2, post_start_x, post_start_x + 0.5, x_end],
        ["-3 s", "-1.5 s", "Bout\nstart", "Start →\n1st lick", "First\nlick", "1st → last lick\n(normalized)", "Last\nlick", "+1.5 s", "+3 s"]
    )

    ax.set_xlabel("Average bout timeline", fontsize=12)
    ax.set_ylabel("Spike activity (z-score)", fontsize=12)
    ax.set_title("Average warped valid bout by depth band", fontsize=15, pad=18)

    ax.grid(alpha=0.18, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98),
               ncol=6, frameon=False, fontsize=10.5)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.show()


# Add remaining plot functions: plot_mean_activity and plot_single_bouts
# These would follow a similar pattern...
