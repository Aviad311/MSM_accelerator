from field import p
import op_counter


# ------------------------------------------------------------
# Golden / Validation affine ops (NORMAL domain)
# Uses Python % freely (this file is NOT for RTL)
# BUT: counts field ops in op_counter for fair comparison.
# ------------------------------------------------------------

def n_add(a: int, b: int) -> int:
    op_counter.field_add_count += 1
    return (a + b) % p

def n_sub(a: int, b: int) -> int:
    op_counter.field_sub_count += 1
    return (a - b) % p

def n_mul(a: int, b: int) -> int:
    op_counter.field_mul_count += 1
    return (a * b) % p

def n_inv(a: int) -> int:
    # assumes a != 0
    op_counter.field_inv_count += 1
    return pow(a, p - 2, p)


def is_on_curve(P):
    if P is None:
        return True
    x, y = P
    left = (y * y) % p
    right = (x * x * x + 7) % p
    return (left - right) % p == 0


def affine_add(P, Q):
    op_counter.affine_add_count += 1

    if P is None:
        return Q
    if Q is None:
        return P

    x1, y1 = P
    x2, y2 = Q

    # P + (-P) = O  <=> x1==x2 and y1 + y2 == 0 (mod p)
    if x1 == x2 and n_add(y1, y2) == 0:
        return None

    # Point doubling
    if P == Q:
        # m = (3*x1^2) / (2*y1)
        num = n_mul(3, n_mul(x1, x1))
        den = n_inv(n_mul(2, y1))
    else:
        # m = (y2 - y1) / (x2 - x1)
        num = n_sub(y2, y1)
        den = n_inv(n_sub(x2, x1))

    m = n_mul(num, den)

    # x3 = m^2 - x1 - x2
    m2 = n_mul(m, m)
    x3 = n_sub(n_sub(m2, x1), x2)

    # y3 = m*(x1 - x3) - y1
    y3 = n_sub(n_mul(m, n_sub(x1, x3)), y1)

    return (x3, y3)


def scalar_mul_affine(k, P):
    R = None
    Q = P

    while k > 0:
        if k & 1:
            R = affine_add(R, Q)
        Q = affine_add(Q, Q)
        k >>= 1

    return R


def msm_naive(scalars, points):
    R = None
    for s, P in zip(scalars, points):
        Pi = scalar_mul_affine(s, P)
        R = affine_add(R, Pi)
    return R
