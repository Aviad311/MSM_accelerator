from field import to_mont
from extended_jacobian import (
    EXT_INF,
    extended_add,
    extended_double,
    extended_mixed_add_mont,
    extended_from_affine_mont,
)

# ------------------------------------------------------------
# Split scalar into windows (LSB first)
# ------------------------------------------------------------
def split_scalar_windows(s, w):
    windows = []
    mask = (1 << w) - 1
    while s > 0:
        windows.append(s & mask)
        s >>= w
    return windows


# ------------------------------------------------------------
# Convert affine NORMAL -> affine MONT once (X,Y)
# ------------------------------------------------------------
def convert_points_to_affine_mont(points_aff):
    """
    Convert all affine NORMAL points to affine MONT once: (X,Y).
    """
    return [(to_mont(x), to_mont(y)) for (x, y) in points_aff]


# ------------------------------------------------------------
# Build buckets using MIXED ADD: Extended bucket + affine point
# ------------------------------------------------------------
def build_buckets_extended_mixed(window_values, points_aff_mont, w):
    """
    points_aff_mont: list of (X2,Y2) affine MONT
    buckets: Extended Jacobian MONT
    """
    num_buckets = 1 << w
    buckets = [EXT_INF] * num_buckets

    for idx, b in enumerate(window_values):
        if b == 0:
            continue

        X2, Y2 = points_aff_mont[idx]

        # infinity check by Z==0 (Z is index 2)
        if buckets[b][2] == 0:
            buckets[b] = extended_from_affine_mont(X2, Y2)  # Z=1, W=1
        else:
            buckets[b] = extended_mixed_add_mont(buckets[b], (X2, Y2))

    return buckets


# ------------------------------------------------------------
# Reduce buckets using Pippenger running-sum method
# ------------------------------------------------------------
def reduce_buckets_extended(buckets):
    running = EXT_INF
    result = EXT_INF

    for i in range(len(buckets) - 1, 0, -1):
        if buckets[i][2] != 0:
            running = extended_add(running, buckets[i])
        result = extended_add(result, running)

    return result


# ------------------------------------------------------------
# Shift accumulated result by w bits (w doublings)
# ------------------------------------------------------------
def shift_window_extended(R, w):
    for _ in range(w):
        R = extended_double(R)
    return R


# ------------------------------------------------------------
# MSM Extended (Pippenger style) - Extended Jacobian Montgomery output
# ------------------------------------------------------------
def msm_extended(scalars, points_aff, w=16):
    """
    - points_aff are AFFINE NORMAL inputs (x,y)
    - internal is Extended Jacobian in Montgomery domain
    - returns Extended Jacobian Montgomery point
    """
    window_lists = [split_scalar_windows(s, w) for s in scalars]
    max_windows = max(len(ws) for ws in window_lists) if window_lists else 0

    # Convert points ONCE: affine MONT (X,Y)
    points_aff_mont = convert_points_to_affine_mont(points_aff)

    R = EXT_INF

    # Process windows from MSB to LSB
    for window_idx in reversed(range(max_windows)):

        # Shift accumulated result (except first iteration)
        if window_idx != max_windows - 1:
            R = shift_window_extended(R, w)

        # Collect window values for this window
        window_vals = []
        for ws in window_lists:
            window_vals.append(ws[window_idx] if window_idx < len(ws) else 0)

        # Build buckets with mixed-add
        buckets = build_buckets_extended_mixed(window_vals, points_aff_mont, w)

        # Reduce buckets (Extended+Extended)
        bucket_sum = reduce_buckets_extended(buckets)

        # Accumulate
        R = extended_add(R, bucket_sum)

    return R
