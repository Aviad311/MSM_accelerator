from field import INF, to_mont, ONE_M
from jacobian import jacobian_add, jacobian_double, jacobian_mixed_add_mont

def window_value(k, window_idx, w):
    mask = (1 << w) - 1
    return (k >> (window_idx * w)) & mask

def shift_window(R, w):
    for _ in range(w):
        R = jacobian_double(R)
    return R

def reduce_buckets_pippenger(buckets):
    running = INF
    result = INF
    for i in range(len(buckets) - 1, 0, -1):
        if buckets[i][2] != 0:
            running = jacobian_add(running, buckets[i])
        result = jacobian_add(result, running)
    return result

def affine_points_to_affine_mont(points_aff):
    """
    points_aff: list of (x,y) in NORMAL domain
    returns: list of (X,Y) in Montgomery AFFINE
    """
    pts = []
    for (x, y) in points_aff:
        pts.append((to_mont(x), to_mont(y)))
    return pts

def msm_pippenger_tiled_pingpong_bundle(
    scalars,
    points_aff,
    w=16,
    tile_size=4,
    batch_size=1024,
    scalar_bits=256
):
    """
    HW-like MSM Pippenger with:
    - Tiled windows (tile_size windows per PASS)
    - Streaming in batches (batch_size)
    - Ping/Pong double-buffering of bucket SRAM
    - "Bundle" overlap: build next tile completely, then reduce previous tile
      (keeps correctness and models the memory architecture; not cycle-accurate).

    IMPORTANT:
    - Convert points to Montgomery AFFINE ONCE.
    - Build buckets using MIXED ADD (Jacobian bucket + affine point) to save field muls.

    Returns: Jacobian Montgomery point.
    """
    N = len(scalars)
    if N == 0:
        return INF

    # Convert points ONCE to AFFINE Montgomery (X,Y)
    points_aff_m = affine_points_to_affine_mont(points_aff)

    num_windows = (scalar_bits + w - 1) // w
    num_buckets = 1 << w

    # Build a list of tiles in MSB->LSB order
    tiles = []
    for win_hi in range(num_windows - 1, -1, -tile_size):
        win_lo = max(0, win_hi - tile_size + 1)
        tiles.append((win_hi, win_lo))

    # Two bucket buffers (ping/pong). Each buffer holds a dict: win -> bucket_array
    buf0 = {}
    buf1 = {}
    build_buf = buf0
    reduce_buf = buf1

    def clear_and_alloc_buffer(buf, win_hi, win_lo):
        buf.clear()
        for win in range(win_lo, win_hi + 1):
            buf[win] = [INF] * num_buckets

    def build_tile_into_buffer(buf, win_hi, win_lo):
        active_windows = list(range(win_lo, win_hi + 1))

        for base in range(0, N, batch_size):
            end = min(N, base + batch_size)

            # "DMA batch"
            for i in range(base, end):
                k = scalars[i]
                X2, Y2 = points_aff_m[i]   # <-- pre-converted affine mont (no to_mont here)

                # Update buckets for all windows in this tile
                for win in active_windows:
                    b = window_value(k, win, w)
                    if b == 0:
                        continue

                    Bi = buf[win][b]
                    if Bi[2] == 0:
                        # First point in bucket: store as Jacobian with Z=1
                        buf[win][b] = (X2, Y2, ONE_M)
                    else:
                        # Mixed add: Jacobian bucket + affine base point
                        buf[win][b] = jacobian_mixed_add_mont(Bi, (X2, Y2))

    # Accumulate results in strict MSB->LSB order (this preserves correctness)
    R = INF
    first_window = True

    def reduce_tile_and_accumulate(buf, win_hi, win_lo):
        nonlocal R, first_window
        for win in range(win_hi, win_lo - 1, -1):
            if not first_window:
                R = shift_window(R, w)
            else:
                first_window = False

            bucket_sum = reduce_buckets_pippenger(buf[win])
            R = jacobian_add(R, bucket_sum)

    # ---- Pipeline over tiles with ping/pong ----
    # 1) Build first tile into build_buf
    first_hi, first_lo = tiles[0]
    clear_and_alloc_buffer(build_buf, first_hi, first_lo)
    build_tile_into_buffer(build_buf, first_hi, first_lo)

    # 2) For each next tile: build it into the other buffer, then reduce previous
    prev_hi, prev_lo = first_hi, first_lo
    reduce_buf, build_buf = build_buf, reduce_buf  # previous tile sits in reduce_buf now

    for (cur_hi, cur_lo) in tiles[1:]:
        # Build current tile into build_buf (fresh)
        clear_and_alloc_buffer(build_buf, cur_hi, cur_lo)
        build_tile_into_buffer(build_buf, cur_hi, cur_lo)

        # Now reduce+accumulate the previous tile (which is in reduce_buf)
        reduce_tile_and_accumulate(reduce_buf, prev_hi, prev_lo)

        # Swap roles: current becomes previous for next iteration
        prev_hi, prev_lo = cur_hi, cur_lo
        reduce_buf, build_buf = build_buf, reduce_buf

    # 3) Reduce+accumulate the last tile still pending in reduce_buf
    reduce_tile_and_accumulate(reduce_buf, prev_hi, prev_lo)

    return R
