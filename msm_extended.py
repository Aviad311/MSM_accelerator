from extended_jacobian import (
    to_extended, extended_mixed_add, extended_add, extended_double, EXT_INF
)



def split_scalar_windows(s, w):
    windows = []
    mask = (1 << w) - 1
    while s > 0:
        windows.append(s & mask)
        s >>= w
    return windows


def build_buckets_extended(window_values, points, w):
    num_buckets = 1 << w

    buckets = [EXT_INF] * num_buckets

    for idx, b in enumerate(window_values):
        if b == 0:
            continue
        P_aff = points[idx]

        if buckets[b][2] == 0:
            buckets[b] = to_extended(P_aff)
        else:
            buckets[b] = extended_mixed_add(buckets[b], P_aff)

    return buckets


def reduce_buckets_extended(buckets):
    running = EXT_INF
    result = EXT_INF

    for i in range(len(buckets) - 1, 0, -1):
        if buckets[i][2] != 0:
            running = extended_add(running, buckets[i])

        result = extended_add(result, running)

    return result


def shift_window_extended(R, w):
    for _ in range(w):
        R = extended_double(R)
    return R


def msm_extended(scalars, points, w=16):
    window_lists = [split_scalar_windows(s, w) for s in scalars]
    max_windows = max(len(ws) for ws in window_lists)

    R = EXT_INF

    for window_idx in reversed(range(max_windows)):
        if window_idx != max_windows - 1:
            R = shift_window_extended(R, w)

        window_vals = []
        for ws in window_lists:
            if window_idx < len(ws):
                window_vals.append(ws[window_idx])
            else:
                window_vals.append(0)

        buckets = build_buckets_extended(window_vals, points, w)
        bucket_sum = reduce_buckets_extended(buckets)
        R = extended_add(R, bucket_sum)

    return R