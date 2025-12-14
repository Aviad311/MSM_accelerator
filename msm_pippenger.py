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
# Build buckets for a single window (same as reference)
# ------------------------------------------------------------

def build_buckets_pippenger(window_values, points, w):
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

        if buckets[b] == INF:
            buckets[b] = (P_aff[0], P_aff[1], 1)
        else:
            buckets[b] = jacobian_mixed_add(buckets[b], P_aff)

    return buckets


# ------------------------------------------------------------
# Reduce buckets using Pippenger running-sum method
# ------------------------------------------------------------

def reduce_buckets_pippenger(buckets):
    """
    Pippenger bucket reduction:
        running = 0
        for i = max_bucket .. 1:
            running += bucket[i]
            result  += running
    """
    running = INF
    result = INF

    for i in range(len(buckets) - 1, 0, -1):
        if buckets[i] != INF:
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
# MSM Pippenger (Fast)
# ------------------------------------------------------------

def msm_pippenger(scalars, points, w=16):
    """
    Fast Multi-Scalar Multiplication using Pippenger algorithm.

    - Same windowing as reference
    - Same bucket building
    - Different (cheap) bucket reduction
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
        buckets = build_buckets_pippenger(window_vals, points, w)

        # Reduce buckets (Pippenger)
        bucket_sum = reduce_buckets_pippenger(buckets)

        # Accumulate
        R = jacobian_add(R, bucket_sum)

    return R
