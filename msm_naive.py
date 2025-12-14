from field import p, f_add, f_sub, f_mul, f_inv

# ------------------------------------------------------------
# Affine point operations on secp256k1
# Curve: y^2 = x^3 + 7
# ------------------------------------------------------------

def is_on_curve(P):
    """
    Check if affine point P lies on the curve.
    P = (x, y) or None (point at infinity)
    """
    if P is None:
        return True

    x, y = P
    return (y * y - (x * x * x + 7)) % p == 0


def affine_add(P, Q):
    """
    Affine point addition.
    P, Q are affine points or None (infinity).
    Returns affine point or None.
    """
    if P is None:
        return Q
    if Q is None:
        return P

    x1, y1 = P
    x2, y2 = Q

    # P + (-P) = O
    if x1 == x2 and (y1 + y2) % p == 0:
        return None

    # Point doubling
    if P == Q:
        num = (3 * x1 * x1) % p
        den = f_inv((2 * y1) % p)
    else:
        # General addition
        num = (y2 - y1) % p
        den = f_inv((x2 - x1) % p)

    m = (num * den) % p

    x3 = (m * m - x1 - x2) % p
    y3 = (m * (x1 - x3) - y1) % p

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
