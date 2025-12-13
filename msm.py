from field import INF
from jacobian import jacobian_add, jacobian_double, jacobian_mixed_add

def split_scalar_windows(s, w):
    """
    Split scalar s into windows of size w bits.
    Returns an array where entry i is the window i.
    """
    windows = []
    mask = (1 << w) - 1   # Mask of w bits (e.g. 0xFFFF for w=16)

    while s > 0:
        windows.append(s & mask)
        s >>= w

    return windows

def build_buckets(window_values, points, w):
    """
    Build buckets for a specific window.
    window_values[i] = scalar window of point P_i.
    """
    num_buckets = (1 << w)
    buckets = [INF] * num_buckets

    for idx, b in enumerate(window_values):
        if b == 0:
            continue  # zero → no contribution
        P = points[idx]

        # buckets[b] = buckets[b] + P  (Jacobian + Affine)
        buckets[b] = jacobian_mixed_add(buckets[b], P)

    return buckets

def reduce_buckets(buckets):
    """
    Sum buckets from highest to lowest.
    standard bucket reduction:
        running = INF
        for b in reversed(buckets):
            running = running + b
            result = result + running
    """
    running = INF
    result = INF

    for b in reversed(buckets):
        if b == INF:
            continue

        # running = running + b   (Jacobian + Jacobian)
        running = jacobian_add(running, b)

        # result = result + running
        result = jacobian_add(result, running)

    return result

def shift_window(R, w):
    for _ in range(w):
        R = jacobian_double(R)
    return R

def msm(scalars, points, w=16):
    """
    Multi-Scalar Multiplication:
        sum_i scalars[i] * points[i]
    Using windowed bucket method.
    """

    # Step 1: split each scalar into windows
    window_lists = [split_scalar_windows(s, w) for s in scalars]
    max_windows = max(len(ws) for ws in window_lists)

    R = INF  # running result

    # For each window index
    for window_idx in reversed(range(max_windows)):

        # Step 2: shift by w bits (doubling w times)
        if R != INF:
            R = shift_window(R, w)

        # Build window values for this round
        window_vals = []
        for ws in window_lists:
            if window_idx < len(ws):
                window_vals.append(ws[window_idx])
            else:
                window_vals.append(0)

        # Step 3: bucket accumulation
        buckets = build_buckets(window_vals, points, w)

        # Step 4: reduce buckets
        bucket_sum = reduce_buckets(buckets)

        # Step 5: accumulate into R
        R = jacobian_add(R, bucket_sum)

    return R
