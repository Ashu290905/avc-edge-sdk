import numpy as np
from scipy.signal import find_peaks, peak_prominences
from scipy.ndimage import gaussian_filter1d
from numpy.polynomial.polynomial import polyfit

NUM_SENSORS = 80

def safe_mean(arr, default=0.0):
    arr = np.asarray(arr)
    return float(np.mean(arr)) if arr.size else default

def safe_std(arr, default=0.0):
    arr = np.asarray(arr)
    return float(np.std(arr)) if arr.size else default

def safe_diff_mean(idxs, default=0.0):
    idxs = np.asarray(idxs)
    return float(np.mean(np.diff(idxs))) if idxs.size > 1 else default

def longest_run_of_ones(row):
    if row.size == 0:
        return 0
    diff = np.diff(np.concatenate(([0], row, [0])))
    run_starts = np.where(diff == 1)[0]
    run_ends   = np.where(diff == -1)[0]
    if run_starts.size == 0:
        return 0
    return int(np.max(run_ends - run_starts))

def parse_ir_string(binary_str):
    bits = [int(b) for b in binary_str.strip() if b in ('0', '1')]
    slices = len(bits) // NUM_SENSORS
    if slices == 0:
        return np.zeros((0, NUM_SENSORS), dtype=int)
    return np.array(bits[:slices * NUM_SENSORS]).reshape((slices, NUM_SENSORS))

def extract_features(profile):
    if profile.size == 0:
        return None

    # Basic profiles
    time_profile = np.sum(profile, axis=1)            # height per slice
    vertical_profile = np.sum(profile, axis=0)
    time_profile_smoothed = gaussian_filter1d(time_profile, sigma=1.5)

    # Pixel counts
    total_pixels     = float(np.sum(profile))
    log_total_pixels = float(np.log1p(total_pixels))
    pixels_per_slice = float(total_pixels / (np.count_nonzero(time_profile) + 1e-5))

    # Height metrics
    active_cols = np.where(profile.any(axis=0))[0]
    if active_cols.size == 0:
        max_height = mean_height = 0.0
    else:
        max_height = NUM_SENSORS - np.min(active_cols)
        slice_heights = NUM_SENSORS - np.argmax(profile[:, ::-1], axis=1)
        mean_height = safe_mean(slice_heights)

    # Length
    slice_active  = (time_profile > 0).astype(int)
    length_slices = int(np.sum(slice_active))
    num_slices    = profile.shape[0]

    # Height stats along length
    height_mean_slice = safe_mean(time_profile)
    height_std_slice  = safe_std(time_profile)

    # Axle valleys
    valleys, _ = find_peaks(-time_profile_smoothed, distance=4, prominence=2)
    axle_count = int(len(valleys))
    prominence = peak_prominences(-time_profile_smoothed, valleys)[0] if axle_count > 0 else np.array([])
    valley_prominence_mean = safe_mean(prominence)
    valley_distance_mean   = safe_diff_mean(valleys)

    # Uplift flag: bottom 2 sensors empty beneath a valley
    axle_uplift_flag = 0
    for v in valleys:
        if v < profile.shape[0] and np.all(profile[v, -2:] == 0):
            axle_uplift_flag = 1
            break

    # Gaps & continuity
    gap_count        = int(np.sum((slice_active[:-1] == 1) & (slice_active[1:] == 0))) if slice_active.size > 1 else 0
    continuity_ratio = float(np.sum(slice_active) / len(slice_active)) if slice_active.size > 0 else 0.0

    # Tail taper
    front_height = float(np.max(profile[0]))  if num_slices > 0 else 0.0
    rear_height  = float(np.max(profile[-1])) if num_slices > 1 else 0.0
    tail_taper   = front_height - rear_height

    # Vertical density / symmetry
    vertical_mean = safe_mean(vertical_profile)
    vertical_std  = safe_std(vertical_profile)

    top_half    = np.sum(profile[:, :NUM_SENSORS // 2])
    bottom_half = np.sum(profile[:, NUM_SENSORS // 2:])
    vertical_density_skew = float((bottom_half - top_half) / (top_half + bottom_half + 1e-5))

    left  = np.sum(profile[:, :NUM_SENSORS // 2])
    right = np.sum(profile[:, NUM_SENSORS // 2:])
    symmetry_score = 1.0 - abs(left - right) / (left + right + 1e-5)

    # YOLO‑style
    aspect_ratio     = float(height_mean_slice / (max_height + 1e-5))
    body_compactness = float(total_pixels / (length_slices * height_mean_slice + 1e-5))
    central_bias     = abs(symmetry_score - 1.0)
    silhouette_slope = float(tail_taper / (length_slices + 1e-5))

    # Structural (cab/gap/body)
    gap_threshold = height_mean_slice * 0.25
    cab_length = int(np.argmax(time_profile > gap_threshold) + 1) if np.any(time_profile > gap_threshold) else 0

    gap_presence = 0
    for i in range(cab_length + 3, len(time_profile) - 3):
        if (time_profile[i] < gap_threshold and
            time_profile[i - 1] > gap_threshold and
            time_profile[i + 1] > gap_threshold):
            gap_presence = 1
            break

    rear = profile[cab_length + 3:]
    if rear.shape[0] >= 5:
        rear_widths  = np.sum(rear, axis=1)
        rear_heights = NUM_SENSORS - np.argmax(rear[:, ::-1], axis=1)
        body_rectangularity_rear = 1.0 / (safe_std(rear_widths) + safe_std(rear_heights) + 1e-5)
    else:
        body_rectangularity_rear = 0.0

    smoothed = np.convolve(time_profile, np.ones(5)/5, mode='same')
    segments, _ = find_peaks(smoothed, prominence=5, distance=6)
    body_segments = int(len(segments))

    # Roof line & Jeep/SUV cues
    roof_line   = []
    bottom_line = []
    for row in profile:
        idx_top = np.argmax(row == 1)
        has_top = (row.size > 0 and row[idx_top] == 1)
        roof_line.append(idx_top if has_top else NUM_SENSORS)
        if row.any():
            idx_bottom_from_top = NUM_SENSORS - 1 - np.argmax(row[::-1] == 1)
            bottom_line.append(idx_bottom_from_top)
        else:
            bottom_line.append(-1)

    roof_line   = np.array(roof_line, dtype=float)
    bottom_line = np.array(bottom_line, dtype=float)

    if roof_line.size:
        roof_smooth = gaussian_filter1d(roof_line, sigma=2)
        roof_range  = float(np.max(roof_smooth) - np.min(roof_smooth))
        roof_std    = safe_std(roof_smooth)

        diffs            = np.abs(np.diff(roof_smooth))
        roof_flat_ratio  = float(np.mean(diffs < 1.0)) if diffs.size else 1.0
        window           = min(10, diffs.size) if diffs.size else 1
        roof_front_slope = safe_mean(diffs[:window])
        roof_rear_slope  = safe_mean(diffs[-window:]) if diffs.size else 0.0

        peak_idx        = int(np.argmin(roof_smooth))  # tallest point (smallest index)
        rise_pixels     = roof_smooth[0] - roof_smooth[peak_idx]
        roof_grad_ratio = float(rise_pixels / (peak_idx + 1e-5))
        roof_max_slope  = float(np.max(-np.diff(roof_smooth))) if diffs.size else 0.0
        roof_jaggedness = safe_mean(np.abs(np.diff(roof_smooth)))

        if num_slices >= 2:
            last_sum  = np.sum(profile[-1])
            prev_sum  = np.sum(profile[-2])
            last_peak = np.argmax(profile[-1][::-1])
            prev_peak = np.argmax(profile[-2][::-1])
            rear_bump_flag = int(last_sum > 1.25 * prev_sum or last_peak < prev_peak - 2)
        else:
            rear_bump_flag = 0

        if roof_smooth.size > 2:
            c0, c1, c2 = polyfit(np.arange(len(roof_smooth)), roof_smooth, 2)
            roof_curvature = float(c2)
        else:
            roof_curvature = 0.0
    else:
        roof_range = roof_std = roof_flat_ratio = 0.0
        roof_front_slope = roof_rear_slope = 0.0
        roof_grad_ratio = roof_max_slope = roof_jaggedness = 0.0
        rear_bump_flag = 0
        roof_curvature = 0.0

    # Global rectangularity & contiguity
    if slice_active.any():
        first_slice = int(np.argmax(slice_active == 1))
        last_slice  = int(len(slice_active) - 1 - np.argmax(slice_active[::-1] == 1))
        width_bb    = max(1, last_slice - first_slice + 1)

        top_series = roof_line[first_slice:last_slice+1]
        bot_series = bottom_line[first_slice:last_slice+1]
        valid = bot_series >= 0
        if np.any(valid):
            heights_est = bot_series[valid] - top_series[valid] + 1.0
            height_bb   = max(1.0, float(np.max(heights_est)))
        else:
            heights_est = np.array([1.0], dtype=float)
            height_bb   = 1.0
        area_bb = width_bb * height_bb
    else:
        width_bb = 1
        height_bb = 1.0
        area_bb = 1.0
        heights_est = np.array([1.0], dtype=float)

    rect_fill_ratio_global = float(total_pixels / (area_bb + 1e-5))

    slice_contiguities = []
    for row in profile:
        ones = int(np.sum(row))
        if ones <= 0:
            slice_contiguities.append(0.0)
        else:
            lrun = longest_run_of_ones(row)
            slice_contiguities.append(float(lrun / (ones + 1e-5)))

    slice_contiguity_mean = safe_mean(np.asarray(slice_contiguities))
    slice_contiguity_std  = safe_std(np.asarray(slice_contiguities))

    # >>> NEW size features (absolute canvas coverage)
    canvas_area        = float(num_slices * NUM_SENSORS) if num_slices > 0 else 1.0
    canvas_fill_ratio  = float(total_pixels / (canvas_area + 1e-5))
    height_norm        = float(height_mean_slice / NUM_SENSORS)
    length_norm        = float(length_slices / (num_slices + 1e-5))
    size_strength      = height_norm * length_norm  # larger & longer => bigger value

    # Final feature vector
    features = [
        # Core shape
        max_height, mean_height, length_slices,
        height_mean_slice, height_std_slice,

        # Axles
        axle_count, valley_prominence_mean, valley_distance_mean, axle_uplift_flag,

        # Gaps & continuity
        gap_count, continuity_ratio, tail_taper,

        # Vertical distribution
        vertical_mean, vertical_std, vertical_density_skew, symmetry_score,

        # Pixel counts
        total_pixels, log_total_pixels, pixels_per_slice,

        # YOLO-ish
        aspect_ratio, body_compactness, central_bias, silhouette_slope,

        # Structural
        cab_length, gap_presence, body_rectangularity_rear, body_segments,

        # Roof stats
        roof_range, roof_std, roof_flat_ratio,
        roof_front_slope, roof_rear_slope, roof_curvature,
        roof_grad_ratio, roof_max_slope, roof_jaggedness, rear_bump_flag,

        # Global rectangularity & contiguity
        rect_fill_ratio_global, slice_contiguity_mean, slice_contiguity_std,

        # >>> NEW absolute size cues
        canvas_fill_ratio, size_strength
    ]

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).tolist()
    return features
