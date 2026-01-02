# ==========================================================
#   Field arithmetic for secp256k1  (GF(p))
#   ASIC-style reference model (no Python % in add/sub/mul/neg)
# ==========================================================
import op_counter

# secp256k1 prime:
# p = 2^256 - 2^32 - 977
p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F

MASK_256 = (1 << 256) - 1

# ----------------------------------------------------------
#  Helper: final canonicalization into [0, p-1]
#  Hardware style: a few conditional subtracts (no division)
# ----------------------------------------------------------
def _canon(x: int) -> int:
    # x is non-negative and not too huge (after our reductions)
    # do a small, fixed number of corrections
    if x >= p:
        x -= p
    if x >= p:
        x -= p
    return x

# ----------------------------------------------------------
#  Addition / subtraction / negation (cheap in hardware)
# ----------------------------------------------------------
def f_add(a: int, b: int) -> int:
    # Field addition: s = a+b; if s>=p => s-=p
    op_counter.field_add_count += 1
    s = a + b
    if s >= p:
        s -= p
    return s

def f_sub(a: int, b: int) -> int:
    # Field subtraction: d = a-b; if borrow => d+=p
    op_counter.field_sub_count += 1
    d = a - b
    if d < 0:
        d += p
    return d

def f_neg(a: int) -> int:
    # Field negation: 0 if a==0 else p-a
    op_counter.field_neg_count += 1
    return 0 if a == 0 else (p - a)

# ----------------------------------------------------------
#  Fast reduction for secp256k1
#  Uses: 2^256 ≡ 2^32 + 977 (mod p)
#
#  For a 512-bit t = lo + hi*2^256,
#  t ≡ lo + hi*(2^32 + 977) (mod p)
# ----------------------------------------------------------
def red_secp256k1_512(t: int) -> int:
    lo = t & MASK_256
    hi = t >> 256  # up to 256 bits

    # x = lo + (hi << 32) + hi*977
    x = lo + (hi << 32)

    # hi*977 = hi*(512 + 256 + 128 + 64 + 16 + 1)
    x += (hi << 9) + (hi << 8) + (hi << 7) + (hi << 6) + (hi << 4) + hi

    # x may still be above 2^256; fold once more using same rule:
    lo2 = x & MASK_256
    hi2 = x >> 256

    # x2 = lo2 + hi2*(2^32 + 977)
    x2 = lo2 + (hi2 << 32)
    x2 += (hi2 << 9) + (hi2 << 8) + (hi2 << 7) + (hi2 << 6) + (hi2 << 4) + hi2

    # Final correction to [0, p-1]
    return _canon(x2)

# ----------------------------------------------------------
#  Multiplication (expensive): multiply then special reduction
# ----------------------------------------------------------
def f_mul(a: int, b: int) -> int:
    op_counter.field_mul_count += 1
    t = a * b  # conceptually 256x256 -> 512-bit product
    return red_secp256k1_512(t)

# ----------------------------------------------------------
#  Inversion (still expensive)
#  Kept as pow(), because in hardware you'll implement separately (FSM)
# ----------------------------------------------------------
def f_inv(a: int) -> int:
    op_counter.field_inv_count += 1
    # This is still a modular exponentiation; expensive in hardware.
    # Model keeps it correct; you can later swap for an ASIC-style exp engine model.
    return pow(a, p - 2, p)

# ----------------------------------------------------------
#  Point at infinity (Jacobian representation)
# ----------------------------------------------------------
INF = (1, 1, 0)
