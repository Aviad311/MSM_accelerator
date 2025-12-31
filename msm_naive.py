from field import p, f_add, f_sub, f_mul, f_inv, f_neg
import op_counter


# ------------------------------------------------------------
# Affine point operations on secp256k1
# Curve: y^2 = x^3 + 7
# ------------------------------------------------------------

from field import p  # רק p

def is_on_curve(P):
    """
    Check if affine point P lies on the curve (NO counter increment).
    Curve: y^2 = x^3 + 7 over GF(p)
    P = (x, y) or None (point at infinity)
    """
    if P is None:
        return True

    x, y = P

    left = (y * y) % p
    right = (x * x * x + 7) % p

    return (left - right) % p == 0



def affine_add(P, Q):
    """
    Affine point addition.
    P, Q are affine points or None (infinity).
    Returns affine point or None.
    """
    op_counter.affine_add_count += 1

    if P is None:
        return Q
    if Q is None:
        return P

    x1, y1 = P
    x2, y2 = Q

    # P + (-P) = O  <=> x1==x2 and y1 + y2 == 0 (mod p)
    if x1 == x2 and f_add(y1, y2) == 0:
        return None

    # Point doubling
    if P == Q:
        # m = (3*x1^2) / (2*y1)
        num = f_mul(3, f_mul(x1, x1))
        den = f_inv(f_mul(2, y1))
    else:
        # m = (y2 - y1) / (x2 - x1)
        num = f_sub(y2, y1)
        den = f_inv(f_sub(x2, x1))

    m = f_mul(num, den)

    # x3 = m^2 - x1 - x2
    m2 = f_mul(m, m)
    x3 = f_sub(f_sub(m2, x1), x2)

    # y3 = m*(x1 - x3) - y1
    y3 = f_sub(f_mul(m, f_sub(x1, x3)), y1)

    return (x3, y3)


# ------------------------------------------------------------
# Naive scalar multiplication (double-and-add)
# ------------------------------------------------------------

def scalar_mul_affine(k, P):
    """
    Compute k * P using naive double-and-add.
    P is affine, result is affine.
    """
    R = None
    Q = P

    while k > 0:
        if k & 1:
            R = affine_add(R, Q)
        Q = affine_add(Q, Q)
        k >>= 1

    return R


# ------------------------------------------------------------
# Naive MSM (Golden)
# ------------------------------------------------------------

def msm_naive(scalars, points):
    """
    Naive Multi-Scalar Multiplication:
        sum_i scalars[i] * points[i]

    - Fully affine
    - Fully naive
    - Very slow
    - Golden reference
    """
    R = None

    for s, P in zip(scalars, points):
        Pi = scalar_mul_affine(s, P)
        R = affine_add(R, Pi)

    return R
