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
    return [(to_mont(x), to_mont(y), ONE_M) for (x, y) in points_aff]


# ------------------------------------------------------------
# Build buckets for a single window (Reference version)
# ------------------------------------------------------------
def build_buckets_reference(window_values, points_jac_m, w):
    """
    bucket[i] = sum of points whose window value == i
    Points are already Jacobian Montgomery.
    Buckets stored in Jacobian Montgomery.
    """
    num_buckets = 1 << w
    buckets = [INF] * num_buckets

    for idx, b in enumerate(window_values):
        if b == 0:
            continue

        Pj = points_jac_m[idx]

        # First contribution
        if buckets[b][2] == 0:
            buckets[b] = Pj
        else:
            buckets[b] = jacobian_add(buckets[b], Pj)

    return buckets


# ------------------------------------------------------------
# Reduce buckets with explicit weights (Golden)
# ------------------------------------------------------------
def reduce_buckets_reference(buckets):
    """
    Reference reduction:
        sum_{i=1..} i * bucket[i]
    Implemented with repeated addition (slow but correct).
    Montgomery Jacobian throughout.
    """
    result = INF

    for i in range(1, len(buckets)):
        if buckets[i][2] == 0:
            continue

        temp = buckets[i]
        for _ in range(i - 1):
            temp = jacobian_add(temp, buckets[i])

        result = jacobian_add(result, temp)

    return result


# ------------------------------------------------------------
# Shift accumulated result by w bits (w doublings)
# ------------------------------------------------------------
def shift_window(R, w):
    for _ in range(w):
        R = jacobian_double(R)
    return R


# ------------------------------------------------------------
# MSM Reference (Golden Model) - Montgomery Jacobian output
# ------------------------------------------------------------
def msm_reference(scalars, points_aff, w=16):
    """
    Reference MSM:
    - windowing + buckets
    - explicit weights (slow but correct)
    - points_aff are affine NORMAL inputs
    - returns Jacobian Montgomery point
    """

    window_lists = [split_scalar_windows(s, w) for s in scalars]
    max_windows = max(len(ws) for ws in window_lists) if window_lists else 0

    # Convert points ONCE
    points_jac_m = affine_points_to_jacobian_mont(points_aff)

    R = INF

    for window_idx in reversed(range(max_windows)):

        if window_idx != max_windows - 1:
            R = shift_window(R, w)

        window_vals = []
        for ws in window_lists:
            window_vals.append(ws[window_idx] if window_idx < len(ws) else 0)

        buckets = build_buckets_reference(window_vals, points_jac_m, w)
        bucket_sum = reduce_buckets_reference(buckets)
        R = jacobian_add(R, bucket_sum)

    return R
