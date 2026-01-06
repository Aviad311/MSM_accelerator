from field import (
    f_add, f_sub, f_mul, f_inv, f_neg,
    to_mont, from_mont,
    p, INF, ONE_M
)
import op_counter


# ----------------------------------------------------------
#  Small constant multiplies in Montgomery domain
#  (avoid f_mul with non-Montgomery constants)
# ----------------------------------------------------------
def mul2(a):
    return f_add(a, a)

def mul3(a):
    return f_add(a, mul2(a))

def mul4(a):
    return mul2(mul2(a))

def mul8(a):
    return mul2(mul4(a))


# ----------------------------------------------------------
#  Jacobian Point Doubling (a=0) - Montgomery-native
# ----------------------------------------------------------
def jacobian_double(P):
    op_counter.jacobian_double_count += 1
    X1, Y1, Z1 = P

    if Z1 == 0 or Y1 == 0:
        return INF

    # S = 4 * X1 * Y1^2
    Y1_sq = f_mul(Y1, Y1)            # Y1^2
    S = mul4(f_mul(X1, Y1_sq))       # 4*X1*Y1^2

    # M = 3 * X1^2
    X1_sq = f_mul(X1, X1)            # X1^2
    M = mul3(X1_sq)                  # 3*X1^2

    # X3 = M^2 - 2*S
    X3 = f_sub(f_mul(M, M), mul2(S))

    # Y3 = M*(S - X3) - 8*(Y1^2)^2
    Y1_sq_sq = f_mul(Y1_sq, Y1_sq)
    Y3 = f_sub(
        f_mul(M, f_sub(S, X3)),
        mul8(Y1_sq_sq)
    )

    # Z3 = 2 * Y1 * Z1
    Z3 = f_mul(mul2(Y1), Z1)

    return (X3, Y3, Z3)


# ----------------------------------------------------------
#  Mixed Addition (Jacobian P + Affine Q), Z2 = 1
#
#  Montgomery-native notes:
#  - P is Montgomery Jacobian (X1,Y1,Z1 are Montgomery)
#  - Q is provided as AFFINE NORMAL (x2,y2) by default,
#    and we convert it ONCE here to Montgomery.
#
#  For ASIC: better to store bases already in Montgomery,
#  then remove these to_mont calls.
# ----------------------------------------------------------
def jacobian_mixed_add(P, Q_aff_normal):
    op_counter.jacobian_mixed_add_count += 1
    X1, Y1, Z1 = P
    x2, y2 = Q_aff_normal  # normal affine

    # Convert affine inputs once
    X2 = to_mont(x2)
    Y2 = to_mont(y2)
    Z2 = ONE_M  # 1 in Montgomery

    if Z1 == 0:
        return (X2, Y2, ONE_M)

    Z1_sq = f_mul(Z1, Z1)         # Z1^2
    U2 = f_mul(X2, Z1_sq)         # x2 * Z1^2

    Z1_cu = f_mul(Z1_sq, Z1)      # Z1^3
    S2 = f_mul(Y2, Z1_cu)         # y2 * Z1^3

    if U2 == X1:
        if S2 != Y1:
            return INF
        return jacobian_double(P)

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

    return (X3, Y3, Z3)


# ----------------------------------------------------------
#  Jacobian Addition (P + Q) - Montgomery-native
# ----------------------------------------------------------
def jacobian_add(P, Q):
    op_counter.jacobian_add_count += 1

    X1, Y1, Z1 = P
    X2, Y2, Z2 = Q

    if Z1 == 0:
        return Q
    if Z2 == 0:
        return P

    Z2_sq = f_mul(Z2, Z2)
    U1 = f_mul(X1, Z2_sq)

    Z1_sq = f_mul(Z1, Z1)
    U2 = f_mul(X2, Z1_sq)

    Z2_cu = f_mul(Z2_sq, Z2)
    S1 = f_mul(Y1, Z2_cu)

    Z1_cu = f_mul(Z1_sq, Z1)
    S2 = f_mul(Y2, Z1_cu)

    if U1 == U2:
        if S1 != S2:
            return INF
        return jacobian_double(P)

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

    return (X3, Y3, Z3)


# ----------------------------------------------------------
#  Jacobian to Affine Conversion (returns NORMAL affine)
#  Input P is Montgomery Jacobian.
# ----------------------------------------------------------
def jacobian_to_affine(P):
    X, Y, Z = P
    if Z == 0:
        return None

    # Zinv in Montgomery domain
    Zinv = f_inv(Z)
    Zinv2 = f_mul(Zinv, Zinv)
    Zinv3 = f_mul(Zinv2, Zinv)

    xM = f_mul(X, Zinv2)
    yM = f_mul(Y, Zinv3)

    # Convert back to normal domain for output
    return (from_mont(xM), from_mont(yM))
