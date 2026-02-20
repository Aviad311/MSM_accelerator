# ==========================================================
#   Field arithmetic for secp256k1  (GF(p))
#   ASIC-style reference model - Montgomery-native datapath
#
#   IMPORTANT:
#   - All field elements are represented in Montgomery domain by default.
#     i.e., aM = a * R mod p, with R = 2^256.
#   - f_add/f_sub/f_neg work the same in Montgomery domain.
#   - f_mul is Montgomery multiply ONLY (no conversions inside).
#   - Use to_mont()/from_mont() only at boundaries (I/O).
# ==========================================================
import op_counter

# secp256k1 prime: p = 2^256 - 2^32 - 977
p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F

MASK_256 = (1 << 256) - 1
R = 1 << 256  # Montgomery radix


# ----------------------------------------------------------
#  Helper: canonicalization into [0, p-1]
#  Hardware style: a few conditional subtracts (no division)
# ----------------------------------------------------------
def _canon(x: int) -> int:
    if x >= p:
        x -= p
    if x >= p:
        x -= p
    return x


# ----------------------------------------------------------
#  Addition / subtraction / negation (cheap in hardware)
#  (Valid both in normal and Montgomery domains)
# ----------------------------------------------------------
def f_add(a: int, b: int) -> int:
    op_counter.field_add_count += 1
    s = a + b
    if s >= p:
        s -= p
    return s

def f_sub(a: int, b: int) -> int:
    op_counter.field_sub_count += 1
    d = a - b
    if d < 0:
        d += p
    return d

def f_neg(a: int) -> int:
    op_counter.field_neg_count += 1
    return 0 if a == 0 else (p - a)


# ----------------------------------------------------------
#  Montgomery constants (INIT-TIME ONLY)
#  In RTL you will hardcode these constants.
#
#  NPRIME = -p^{-1} mod R
#  R2     = R^2 mod p
#  ONE_M  = 1 in Montgomery domain = R mod p
# ----------------------------------------------------------
def _mont_params():
    # init-time; not part of datapath
    pinv = pow(p, -1, R)              # p^{-1} mod R
    nprime = (-pinv) & MASK_256       # -p^{-1} mod R
    r2 = (R * R) % p                  # R^2 mod p  (init-time only)
    return nprime, r2

NPRIME, R2 = _mont_params()

# ONE_M = R mod p. For secp256k1: 2^256 ≡ 2^32 + 977 (mod p)
ONE_M = (1 << 32) + 977
ZERO  = 0


# ----------------------------------------------------------
#  Montgomery reduction: REDC(t) = t * R^{-1} mod p
#
#  Datapath ops only: mask, shift, mul, add, conditional subtract
#  t can be up to ~512 bits here.
# ----------------------------------------------------------
def mont_red(t: int) -> int:
    # m = (t * NPRIME) mod R
    m = (t * NPRIME) & MASK_256

    # u = (t + m*p) / R   (exact because (t + m*p) divisible by R)
    u = (t + m * p) >> 256

    # final conditional subtract
    if u >= p:
        u -= p
    return u


# ----------------------------------------------------------
#  Multiplication (expensive): Montgomery multiply
#
#  Inputs/outputs are Montgomery domain values.
#  If aM=aR and bM=bR then f_mul(aM,bM) = abR (still Montgomery).
# ----------------------------------------------------------
def f_mul(a: int, b: int) -> int:
    op_counter.field_mul_count += 1
    return mont_red(a * b)


# ----------------------------------------------------------
#  Boundary conversions (ONLY use at I/O boundaries)
# ----------------------------------------------------------
def to_mont(a: int) -> int:
    """
    Convert normal a -> aM = a*R mod p.
    Implemented by: MontMul(a, R^2) = a*R^2*R^{-1} = a*R (mod p)
    Note: 'a' here is in normal domain; this is fine for this software model.
    """
    if a >= p:
        a = _canon(a)
    return f_mul(a, R2)

def from_mont(aM: int) -> int:
    """
    Convert Montgomery aM -> normal a.
    a = MontRed(aM) because aM = a*R  => aM*R^{-1} = a
    """
    return mont_red(aM)


# ----------------------------------------------------------
#  Inversion (still expensive)
#  Return value is also in Montgomery domain.
#
#  Reference model uses pow() in normal domain, then converts back.
#  RTL: implement exponentiation/EEA engine operating on Montgomery values.
# ----------------------------------------------------------
def f_inv(aM: int) -> int:
    op_counter.field_inv_count += 1

    # convert out (normal), invert, convert back
    a = from_mont(aM)
    ainv = pow(a, p - 2, p)
    return to_mont(ainv)


# ----------------------------------------------------------
#  Point at infinity (Jacobian representation)
#  Z=0 encodes INF.
#  NOTE: When the rest of your Jacobian code becomes Montgomery-native,
#        you'll likely want INF = (ONE_M, ONE_M, 0).
#        We'll update it when we edit jacobian.py.
# ----------------------------------------------------------
INF = (ONE_M, ONE_M, 0)

