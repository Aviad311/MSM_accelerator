from field import INF, to_mont, ONE_M
from jacobian import (
    jacobian_add,
    jacobian_double,
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
# Convert affine NORMAL points -> Jacobian MONT points (once)
# ------------------------------------------------------------
def affine_points_to_jacobian_mont(points_aff):
    """
    points_aff: list of (x,y) in NORMAL domain
    returns: list of (X,Y,Z) in Montgomery Jacobian
    """
    pts = []
    for (x, y) in points_aff:
        pts.append((to_mont(x), to_mont(y), ONE_M))
    return pts


# ------------------------------------------------------------
# Build buckets for a single window (Montgomery Jacobian points)
# ------------------------------------------------------------
def build_buckets_pippenger(window_values, points_jac_m, w):
    """
    bucket[i] = sum of points whose window value == i
    Points are already in Jacobian Montgomery form.
    Buckets stored in Jacobian Montgomery.
    """
    num_buckets = 1 << w
    buckets = [INF] * num_buckets

    for idx, b in enumerate(window_values):
        if b == 0:
            continue

        Pj = points_jac_m[idx]

        if buckets[b][2] == 0:     # INF check by Z==0 (safer than tuple compare)
            buckets[b] = Pj
        else:
            buckets[b] = jacobian_add(buckets[b], Pj)

    return buckets


# ------------------------------------------------------------
# Reduce buckets using Pippenger running-sum method
# ------------------------------------------------------------
def reduce_buckets_pippenger(buckets):
    running = INF
    result = INF

    for i in range(len(buckets) - 1, 0, -1):
        if buckets[i][2] != 0:
            running = jacobian_add(running, buckets[i])
        result = jacobian_add(result, running)

    return result


# ------------------------------------------------------------
# Shift accumulated result by w bits (w doublings)
# ------------------------------------------------------------
def shift_window(R, w):
    for _ in range(w):
        R = jacobian_double(R)
    return R


# ------------------------------------------------------------
# MSM Pippenger (Fast) - Montgomery-native output (Jacobian)
# ------------------------------------------------------------
def msm_pippenger(scalars, points_aff, w=16):
    """
    Fast MSM using Pippenger.
    - points_aff are NORMAL affine inputs (x,y)
    - internal computations are Montgomery Jacobian
    - returns Jacobian Montgomery point
    """

    # Convert points ONCE (huge win)
    points_jac_m = affine_points_to_jacobian_mont(points_aff)

    # Split scalars into windows
    window_lists = [split_scalar_windows(s, w) for s in scalars]
    max_windows = max(len(ws) for ws in window_lists) if window_lists else 0

    R = INF

    # Process windows from MSB to LSB
    for window_idx in reversed(range(max_windows)):

        # Shift accumulated result (except first iteration)
        if window_idx != max_windows - 1:
            R = shift_window(R, w)

        # Collect window values for this window
        window_vals = []
        for ws in window_lists:
            window_vals.append(ws[window_idx] if window_idx < len(ws) else 0)

        # Build buckets
        buckets = build_buckets_pippenger(window_vals, points_jac_m, w)

        # Reduce buckets (Pippenger)
        bucket_sum = reduce_buckets_pippenger(buckets)

        # Accumulate
        R = jacobian_add(R, bucket_sum)

    return R
