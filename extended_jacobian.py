from field import f_add, f_sub, f_mul, f_inv, p
import op_counter

# Extended Jacobian representation (X, Y, Z, W) where W = Z^2
EXT_INF = (1, 1, 0, 0)


def to_extended(P_aff):
    """
    Convert an affine point (x, y) to extended Jacobian (X, Y, Z, W).
    For affine: Z = 1, so W = Z^2 = 1.
    """
    if P_aff is None:
        return EXT_INF
    return (P_aff[0] % p, P_aff[1] % p, 1, 1)


from field import p  # רק p, בלי f_*

def extended_to_affine(P):
    """
    Convert extended Jacobian point (X, Y, Z, W) to affine (x, y)
    WITHOUT counting field operations.
    """
    X, Y, Z, W = P
    if Z == 0:
        return None

    Z_inv = pow(Z, p - 2, p)  # inversion without counting
    Z_inv_sq = (Z_inv * Z_inv) % p
    Z_inv_cu = (Z_inv_sq * Z_inv) % p

    x = (X * Z_inv_sq) % p
    y = (Y * Z_inv_cu) % p
    return (x, y)


def extended_double(P):
    op_counter.extended_double_count += 1

    X1, Y1, Z1, W1 = P

    if Z1 == 0 or Y1 == 0:
        return EXT_INF

    # Same formulas as Jacobian doubling, plus W3 = Z3^2
    Y1_sq = f_mul(Y1, Y1)                 # Y1^2
    S = f_mul(4, f_mul(X1, Y1_sq))        # 4*X1*Y1^2
    X1_sq = f_mul(X1, X1)                 # X1^2
    M = f_mul(3, X1_sq)                   # 3*X1^2 (a=0)

    X3 = f_sub(f_mul(M, M), f_mul(2, S))  # M^2 - 2S

    Y1_sq_sq = f_mul(Y1_sq, Y1_sq)        # (Y1^2)^2
    Y3 = f_sub(f_mul(M, f_sub(S, X3)), f_mul(8, Y1_sq_sq))

    Z3 = f_mul(2, f_mul(Y1, Z1))          # 2*Y1*Z1
    W3 = f_mul(Z3, Z3)                    # Z3^2

    return (X3, Y3, Z3, W3)


def extended_mixed_add(P, Q_aff):
    """
    Extended Jacobian P + affine Q (x2, y2), using W1 = Z1^2.
    """
    op_counter.extended_mixed_add_count += 1

    X1, Y1, Z1, W1 = P
    x2, y2 = Q_aff

    if Z1 == 0:
        return to_extended(Q_aff)

    # U2 = x2 * Z1^2 = x2 * W1
    U2 = f_mul(x2, W1)

    # S2 = y2 * Z1^3 = y2 * (Z1 * Z1^2) = y2 * (Z1 * W1)
    S2 = f_mul(y2, f_mul(Z1, W1))

    if U2 == X1:
        if S2 != Y1:
            return EXT_INF
        return extended_double(P)

    H = f_sub(U2, X1)
    R = f_sub(S2, Y1)

    H_sq = f_mul(H, H)
    H_cu = f_mul(H_sq, H)

    X3 = f_sub(
        f_sub(f_mul(R, R), H_cu),
        f_mul(2, f_mul(X1, H_sq))
    )

    Y3 = f_sub(
        f_mul(R, f_sub(f_mul(X1, H_sq), X3)),
        f_mul(Y1, H_cu)
    )

    Z3 = f_mul(Z1, H)
    W3 = f_mul(Z3, Z3)

    return (X3, Y3, Z3, W3)


def extended_add(P, Q):
    """
    Extended Jacobian addition P + Q (both in (X,Y,Z,W)).
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
    R = f_sub(S2, S1)

    H_sq = f_mul(H, H)
    H_cu = f_mul(H_sq, H)

    X3 = f_sub(
        f_sub(f_mul(R, R), H_cu),
        f_mul(2, f_mul(U1, H_sq))
    )

    Y3 = f_sub(
        f_mul(R, f_sub(f_mul(U1, H_sq), X3)),
        f_mul(S1, H_cu)
    )

    Z3 = f_mul(f_mul(Z1, Z2), H)
    W3 = f_mul(Z3, Z3)

    return (X3, Y3, Z3, W3)
