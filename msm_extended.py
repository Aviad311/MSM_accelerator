from extended_jacobian import (
    to_extended,          # should convert affine NORMAL -> extended MONT
    extended_add,
    extended_double,
    EXT_INF
)

def split_scalar_windows(s, w):
    windows = []
    mask = (1 << w) - 1
    while s > 0:
        windows.append(s & mask)
        s >>= w
    return windows


def convert_points_to_extended(points_aff):
    """
    Convert all affine NORMAL points to Extended Jacobian (Montgomery domain) once.
    """
    return [to_extended(P) for P in points_aff]


def build_buckets_extended(window_values, points_ext, w):
    """
    points_ext are already Extended (Montgomery domain).
    buckets are Extended (Montgomery domain).
    """
    num_buckets = 1 << w
    buckets = [EXT_INF] * num_buckets

    for idx, b in enumerate(window_values):
        if b == 0:
            continue

        P_ext = points_ext[idx]

        # check infinity by Z==0
        if buckets[b][2] == 0:
            buckets[b] = P_ext
        else:
            buckets[b] = extended_add(buckets[b], P_ext)

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


def msm_extended(scalars, points_aff, w=16):
    """
    - points_aff are AFFINE NORMAL inputs (x,y)
    - internal is Extended Jacobian in Montgomery domain
    - returns Extended Jacobian Montgomery point
    """
    window_lists = [split_scalar_windows(s, w) for s in scalars]
    max_windows = max(len(ws) for ws in window_lists) if window_lists else 0

    # Convert points ONCE
    points_ext = convert_points_to_extended(points_aff)

    R = EXT_INF

    for window_idx in reversed(range(max_windows)):
        if window_idx != max_windows - 1:
            R = shift_window_extended(R, w)

        window_vals = []
        for ws in window_lists:
            window_vals.append(ws[window_idx] if window_idx < len(ws) else 0)

        buckets = build_buckets_extended(window_vals, points_ext, w)
        bucket_sum = reduce_buckets_extended(buckets)
        R = extended_add(R, bucket_sum)

    return R
