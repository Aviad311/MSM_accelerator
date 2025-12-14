from field import INF
from jacobian import (
    jacobian_add,
    jacobian_double,
    jacobian_mixed_add
)

# ------------------------------------------------------------
# Split scalar into windows (LSB first)
# ------------------------------------------------------------

def split_scalar_windows(s, w):
    """
    Split scalar s into windows of size w bits.
    windows[0] = LSB window
    """
    windows = []
    mask = (1 << w) - 1

    while s > 0:
        windows.append(s & mask)
        s >>= w

    return windows


# ------------------------------------------------------------
# Build buckets for a single window (Reference version)
# ------------------------------------------------------------

def build_buckets_reference(window_values, points, w):
    """
    bucket[i] = sum of points whose window value == i
    Points are given in affine form.
    Buckets are stored in Jacobian.
    """
    num_buckets = 1 << w
    buckets = [INF] * num_buckets

    for idx, b in enumerate(window_values):
        if b == 0:
            continue

        P_aff = points[idx]

        # First contribution → affine → Jacobian
        if buckets[b] == INF:
            buckets[b] = (P_aff[0], P_aff[1], 1)
        else:
            # Mixed add: Jacobian + Affine
            buckets[b] = jacobian_mixed_add(buckets[b], P_aff)

    return buckets


# ------------------------------------------------------------
# Reduce buckets with explicit weights (Golden)
# ------------------------------------------------------------

def reduce_buckets_reference(buckets):
    """
    Reference reduction:
        sum_{i=1..} i * bucket[i]
    Implemented with repeated addition (slow but correct).
    """
    result = INF

    for i in range(1, len(buckets)):
        if buckets[i] == INF:
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
# MSM Reference (Golden Model)
# ------------------------------------------------------------

def msm_reference(scalars, points, w=16):
    """
    Reference Multi-Scalar Multiplication:
        sum_i scalars[i] * points[i]

    - Uses windowing
    - Uses buckets
    - Uses explicit weights (slow but correct)
    """

    # Split scalars into windows
    window_lists = [split_scalar_windows(s, w) for s in scalars]
    max_windows = max(len(ws) for ws in window_lists)

    R = INF

    # Process windows from MSB to LSB
    for window_idx in reversed(range(max_windows)):

        # Shift accumulated result (except first iteration)
        if window_idx != max_windows - 1:
            R = shift_window(R, w)

        # Collect window values for this window
        window_vals = []
        for ws in window_lists:
            if window_idx < len(ws):
                window_vals.append(ws[window_idx])
            else:
                window_vals.append(0)

        # Build buckets
        buckets = build_buckets_reference(window_vals, points, w)

        # Reduce buckets (explicit weights)
        bucket_sum = reduce_buckets_reference(buckets)

        # Accumulate into result
        R = jacobian_add(R, bucket_sum)

    return R
