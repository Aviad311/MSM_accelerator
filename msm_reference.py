from field import INF, to_mont, ONE_M
from jacobian import (
    jacobian_add,
    jacobian_double,
    jacobian_mixed_add_mont,
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
# Convert affine NORMAL points -> AFFINE MONT points (once)
# ------------------------------------------------------------
def affine_points_to_affine_mont(points_aff):
    """
    points_aff: list of (x,y) in NORMAL domain
    returns: list of (X,Y) in Montgomery AFFINE
    """
    return [(to_mont(x), to_mont(y)) for (x, y) in points_aff]


# ------------------------------------------------------------
# Build buckets for a single window (Reference version) - MIXED
# ------------------------------------------------------------
def build_buckets_reference(window_values, points_aff_mont, w):
    """
    bucket[i] = sum of points whose window value == i

    Points are AFFINE Montgomery (X2,Y2).
    Buckets stored in Jacobian Montgomery (X,Y,Z).

    Update rule:
      - if bucket empty: bucket = (X2,Y2,1)
      - else: bucket = mixed_add(bucket, (X2,Y2))
    """
    num_buckets = 1 << w
    buckets = [INF] * num_buckets

    for idx, b in enumerate(window_values):
        if b == 0:
            continue

        X2, Y2 = points_aff_mont[idx]

        if buckets[b][2] == 0:
            buckets[b] = (X2, Y2, ONE_M)
        else:
            buckets[b] = jacobian_mixed_add_mont(buckets[b], (X2, Y2))

    return buckets


# ------------------------------------------------------------
# Reduce buckets with explicit weights (Golden)
# ------------------------------------------------------------
def reduce_buckets_reference(buckets):
    """
    Reference reduction:
        sum_{i=1..} i * bucket[i]
    Implemented with repeated addition (slow but correct).

    Buckets are Jacobian Montgomery.
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
    - converts bases ONCE to affine MONT
    - builds buckets with MIXED add to match HW-friendly flow
    - returns Jacobian Montgomery point
    """

    window_lists = [split_scalar_windows(s, w) for s in scalars]
    max_windows = max(len(ws) for ws in window_lists) if window_lists else 0

    # Convert points ONCE to AFFINE Montgomery
    points_aff_mont = affine_points_to_affine_mont(points_aff)

    R = INF

    for window_idx in reversed(range(max_windows)):

        if window_idx != max_windows - 1:
            R = shift_window(R, w)

        window_vals = []
        for ws in window_lists:
            window_vals.append(ws[window_idx] if window_idx < len(ws) else 0)

        buckets = build_buckets_reference(window_vals, points_aff_mont, w)
        bucket_sum = reduce_buckets_reference(buckets)
        R = jacobian_add(R, bucket_sum)

    return R
