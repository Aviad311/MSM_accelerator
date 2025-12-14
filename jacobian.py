from field import f_add, f_sub, f_mul, f_inv, f_neg, p, INF

# ----------------------------------------------------------
#  Jacobian Point Doubling
# ----------------------------------------------------------

def jacobian_double(P):
    X1, Y1, Z1 = P

    if Z1 == 0 or Y1 == 0:
        return INF

    # S = 4 * X1 * Y1^2
    Y1_sq = f_mul(Y1, Y1)                # Y1^2 #1
    S = f_mul(4, f_mul(X1, Y1_sq))       # S = 4X1(Y1^2) #2

    # M = 3 * X1^2   (because a = 0 in secp256k1)
    X1_sq = f_mul(X1, X1) #3
    M = f_mul(3, X1_sq)

    # X3 = M^2 - 2*S
    X3 = f_sub(f_mul(M, M), f_mul(2, S)) #4

    # Y3 = M*(S - X3) - 8*(Y1^2)^2
    Y1_sq_sq = f_mul(Y1_sq, Y1_sq)       # (Y1^2)^2 #5
    Y3 = f_sub(
            f_mul(M, f_sub(S, X3)),  #6
            f_mul(8, Y1_sq_sq)
        )

    # Z3 = 2 * Y1 * Z1
    Z3 = f_mul(2, f_mul(Y1, Z1)) #7

    return (X3 % p, Y3 % p, Z3 % p)

# ----------------------------------------------------------
#  Mixed Addition   (Jacobian P + Affine Q with Z2 = 1)
#  Used for bucket building in MSM
# ----------------------------------------------------------

def jacobian_mixed_add(P, Q):
    X1, Y1, Z1 = P
    x2, y2 = Q   # Affine (Z2 = 1)

    if Z1 == 0:
        return (x2, y2, 1)

    Z1_sq = f_mul(Z1, Z1) #1
    U2 = f_mul(x2, Z1_sq)  #2
    Z1_cu = f_mul(Z1_sq, Z1)  #3
    S2 = f_mul(y2, Z1_cu)   #4

    if U2 == X1:
        if S2 != Y1:
            return INF
        return jacobian_double(P)

    H = f_sub(U2, X1)
    R = f_sub(S2, Y1)

    H_sq = f_mul(H, H)  #5
    H_cu = f_mul(H_sq, H)  #6

    X3 = f_sub(
            f_sub(f_mul(R, R), H_cu),  #7
            f_mul(2, f_mul(X1, H_sq))  #8
        )

    Y3 = f_sub(
            f_mul(R, f_sub(f_mul(X1, H_sq), X3)), #8
            f_mul(Y1, H_cu) #9
        )

    Z3 = f_mul(Z1, H) #10

    return (X3 % p, Y3 % p, Z3 % p)


def jacobian_add(P, Q):

    X1, Y1, Z1 = P
    X2, Y2, Z2 = Q

    if Z1 == 0:
        return Q
    if Z2 == 0:
        return P

    Z2_sq = f_mul(Z2, Z2)  #1
    U1 = f_mul(X1, Z2_sq)  #2

    Z1_sq = f_mul(Z1, Z1) #3
    U2 = f_mul(X2, Z1_sq)  #4

    Z2_cu = f_mul(Z2_sq, Z2) #5
    S1 = f_mul(Y1, Z2_cu)   #6

    Z1_cu = f_mul(Z1_sq, Z1)  #7
    S2 = f_mul(Y2, Z1_cu)    #8

    if U1 == U2:
        if S1 != S2:
            return INF
        return jacobian_double(P)

    H = f_sub(U2, U1)
    R = f_sub(S2, S1)

    H_sq = f_mul(H, H)  #9
    H_cu = f_mul(H_sq, H) #10

    X3 = f_sub(
            f_sub(f_mul(R, R), H_cu),  #11
            f_mul(2, f_mul(U1, H_sq))  #12
         )

    Y3 = f_sub(
            f_mul(R, f_sub(f_mul(U1, H_sq), X3)),  #12
            f_mul(S1, H_cu)  #13
         )

    Z3 = f_mul(f_mul(Z1, Z2), H)  #13 #14

    return (X3 % p, Y3 % p, Z3 % p)
from field import p

def jacobian_to_affine(P):
    """
    Convert Jacobian point to affine coordinates.
    P = (X, Y, Z)
    Returns (x, y) or None if point at infinity.
    """
    X, Y, Z = P

    if Z == 0:
        return None

    Z_inv = pow(Z, p - 2, p)

    x = (X * Z_inv * Z_inv) % p
    y = (Y * Z_inv * Z_inv * Z_inv) % p

    return (x, y)
