from field import (
    f_add, f_sub, f_mul, f_inv,
    to_mont, from_mont,
    p, ONE_M
)
import op_counter

# Extended Jacobian representation (X, Y, Z, W) where W = Z^2
# All coordinates are in Montgomery domain.
EXT_INF = (ONE_M, ONE_M, 0, 0)


# ----------------------------------------------------------
# Small constant multiplies in Montgomery domain
# ----------------------------------------------------------
def mul2(a): return f_add(a, a)
def mul3(a): return f_add(a, mul2(a))
def mul4(a): return mul2(mul2(a))
def mul8(a): return mul2(mul4(a))


def to_extended(P_aff):
    """
    Convert affine NORMAL point (x, y) to extended Jacobian (X, Y, Z, W),
    returning Montgomery domain coordinates.
      Z = 1 (Montgomery => ONE_M)
      W = Z^2 = 1 (Montgomery => ONE_M)
    """
    if P_aff is None:
        return EXT_INF
    x, y = P_aff
    return (to_mont(x), to_mont(y), ONE_M, ONE_M)


def extended_from_affine_mont(X2, Y2):
    """
    Build Extended Jacobian point from affine MONT (X2,Y2) with Z=1, W=1.
    This is the "no-conversion" version for MSM/HW flow.
    """
    return (X2, Y2, ONE_M, ONE_M)


def extended_to_affine(P):
    """
    Convert extended Jacobian (Montgomery) -> affine NORMAL (x,y).
    Uses Montgomery inversion and boundary conversion.
    """
    X, Y, Z, W = P
    if Z == 0:
        return None

    Z_inv = f_inv(Z)              # Montgomery
    Z_inv_sq = f_mul(Z_inv, Z_inv)
    Z_inv_cu = f_mul(Z_inv_sq, Z_inv)

    xM = f_mul(X, Z_inv_sq)
    yM = f_mul(Y, Z_inv_cu)

    return (from_mont(xM), from_mont(yM))


def extended_double(P):
    op_counter.extended_double_count += 1

    X1, Y1, Z1, W1 = P

    if Z1 == 0 or Y1 == 0:
        return EXT_INF

    # Same formulas as Jacobian doubling, plus W3 = Z3^2
    Y1_sq = f_mul(Y1, Y1)                 # Y1^2
    S = mul4(f_mul(X1, Y1_sq))            # 4*X1*Y1^2

    X1_sq = f_mul(X1, X1)                 # X1^2
    M = mul3(X1_sq)                       # 3*X1^2 (a=0)

    X3 = f_sub(f_mul(M, M), mul2(S))      # M^2 - 2S

    Y1_sq_sq = f_mul(Y1_sq, Y1_sq)        # (Y1^2)^2
    Y3 = f_sub(f_mul(M, f_sub(S, X3)), mul8(Y1_sq_sq))

    Z3 = f_mul(mul2(Y1), Z1)              # 2*Y1*Z1
    W3 = f_mul(Z3, Z3)                    # Z3^2

    return (X3, Y3, Z3, W3)


def extended_mixed_add_mont(P, Q_aff_mont):
    """
    Extended Jacobian P + affine Q, where Q is already in MONT domain.

    P: (X1,Y1,Z1,W1)  Extended Montgomery, with W1 = Z1^2
    Q_aff_mont: (X2,Y2) affine Montgomery

    Returns: (X3,Y3,Z3,W3) Extended Montgomery
    """
    op_counter.extended_mixed_add_count += 1

    X1, Y1, Z1, W1 = P
    X2, Y2 = Q_aff_mont

    if Z1 == 0:
        return extended_from_affine_mont(X2, Y2)

    # U2 = X2 * Z1^2 = X2 * W1
    U2 = f_mul(X2, W1)

    # S2 = Y2 * Z1^3 = Y2 * (Z1 * W1)
    S2 = f_mul(Y2, f_mul(Z1, W1))

    if U2 == X1:
        if S2 != Y1:
            return EXT_INF
        return extended_double(P)

    H = f_sub(U2, X1)
    Rr = f_sub(S2, Y1)

    H_sq = f_mul(H, H)
    H_cu = f_mul(H_sq, H)

    X1H2 = f_mul(X1, H_sq)

    X3 = f_sub(
        f_sub(f_mul(Rr, Rr), H_cu),
        mul2(X1H2)
    )

    Y3 = f_sub(
        f_mul(Rr, f_sub(X1H2, X3)),
        f_mul(Y1, H_cu)
    )

    Z3 = f_mul(Z1, H)
    W3 = f_mul(Z3, Z3)

    return (X3, Y3, Z3, W3)


def extended_mixed_add(P, Q_aff):
    """
    Extended Jacobian P + affine Q (x2, y2).
    P is Montgomery extended.
    Q_aff is affine NORMAL; converted once here to Montgomery.

    This is a convenience wrapper.
    For MSM/HW flow, prefer extended_mixed_add_mont with pre-converted points.
    """
    X1, Y1, Z1, W1 = P
    if Z1 == 0:
        return to_extended(Q_aff)

    x2, y2 = Q_aff
    X2 = to_mont(x2)
    Y2 = to_mont(y2)
    return extended_mixed_add_mont(P, (X2, Y2))


def extended_add(P, Q):
    """
    Extended Jacobian addition P + Q (both in (X,Y,Z,W), Montgomery domain).
    """
    op_counter.extended_add_count += 1

    X1, Y1, Z1, W1 = P
    X2, Y2, Z2, W2 = Q

    if Z1 == 0:
        return Q
    if Z2 == 0:
        return P

    # U1 = X1 * Z2^2 = X1 * W2
    U1 = f_mul(X1, W2)
    # U2 = X2 * Z1^2 = X2 * W1
    U2 = f_mul(X2, W1)

    # S1 = Y1 * Z2^3 = Y1 * (Z2 * W2)
    S1 = f_mul(Y1, f_mul(Z2, W2))
    # S2 = Y2 * Z1^3 = Y2 * (Z1 * W1)
    S2 = f_mul(Y2, f_mul(Z1, W1))

    if U1 == U2:
        if S1 != S2:
            return EXT_INF
        return extended_double(P)

    H = f_sub(U2, U1)
    Rr = f_sub(S2, S1)

    H_sq = f_mul(H, H)
    H_cu = f_mul(H_sq, H)

    U1H2 = f_mul(U1, H_sq)

    X3 = f_sub(
        f_sub(f_mul(Rr, Rr), H_cu),
        mul2(U1H2)
    )

    Y3 = f_sub(
        f_mul(Rr, f_sub(U1H2, X3)),
        f_mul(S1, H_cu)
    )

    Z3 = f_mul(f_mul(Z1, Z2), H)
    W3 = f_mul(Z3, Z3)

    return (X3, Y3, Z3, W3)
