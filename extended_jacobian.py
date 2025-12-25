from field import f_add, f_sub, f_mul, f_inv, p
import op_counter  # <--- חשוב מאוד לייבא את המונים

EXT_INF = (1, 1, 0, 0)



def to_extended(P_aff):
    if P_aff is None: return EXT_INF
    return (P_aff[0], P_aff[1], 1, 1)


def extended_to_affine(P):
    X, Y, Z, W = P
    if Z == 0: return None
    Z_inv = f_inv(Z)
    Z_inv_sq = f_mul(Z_inv, Z_inv)
    Z_inv_cu = f_mul(Z_inv_sq, Z_inv)
    return (f_mul(X, Z_inv_sq), f_mul(Y, Z_inv_cu))


# ----------------------------------------------------------
# פעולות אריתמטיות עם מונים פעילים
# ----------------------------------------------------------

def extended_double(P):
    # הנה התיקון: השורה הזו כבר לא בהערה!
    op_counter.extended_double_count += 1

    X1, Y1, Z1, W1 = P

    if Z1 == 0:
        return EXT_INF

    # ... אותו חישוב בדיוק ...
    Y1_sq = f_mul(Y1, Y1)
    S = f_mul(4, f_mul(X1, Y1_sq))
    X1_sq = f_mul(X1, X1)
    M = f_mul(3, X1_sq)

    X3 = f_sub(f_mul(M, M), f_mul(2, S))

    Y1_sq_sq = f_mul(Y1_sq, Y1_sq)
    Y3 = f_sub(f_mul(M, f_sub(S, X3)), f_mul(8, Y1_sq_sq))

    Z3 = f_mul(2, f_mul(Y1, Z1))
    W3 = f_mul(Z3, Z3)

    return (X3, Y3, Z3, W3)


def extended_mixed_add(P, Q_aff):
    # הנה התיקון:
    op_counter.extended_mixed_add_count += 1

    X1, Y1, Z1, W1 = P
    x2, y2 = Q_aff

    if Z1 == 0:
        return to_extended(Q_aff)

    U2 = f_mul(x2, W1)
    S2 = f_mul(y2, f_mul(Z1, W1))

    if U2 == X1:
        if S2 != Y1:
            return EXT_INF
        return extended_double(P)

    H = f_sub(U2, X1)
    R = f_sub(S2, Y1)
    H_sq = f_mul(H, H)
    H_cu = f_mul(H_sq, H)

    X3 = f_sub(f_sub(f_mul(R, R), H_cu), f_mul(2, f_mul(X1, H_sq)))
    Y3 = f_sub(f_mul(R, f_sub(f_mul(X1, H_sq), X3)), f_mul(Y1, H_cu))
    Z3 = f_mul(Z1, H)
    W3 = f_mul(Z3, Z3)

    return (X3, Y3, Z3, W3)


def extended_add(P, Q):
    # הנה התיקון:
    op_counter.extended_add_count += 1

    X1, Y1, Z1, W1 = P
    X2, Y2, Z2, W2 = Q

    if Z1 == 0: return Q
    if Z2 == 0: return P

    U1 = f_mul(X1, W2)
    U2 = f_mul(X2, W1)
    S1 = f_mul(Y1, f_mul(Z2, W2))
    S2 = f_mul(Y2, f_mul(Z1, W1))

    if U1 == U2:
        if S1 != S2: return EXT_INF
        return extended_double(P)

    H = f_sub(U2, U1)
    R = f_sub(S2, S1)
    H_sq = f_mul(H, H)
    H_cu = f_mul(H_sq, H)

    X3 = f_sub(f_sub(f_mul(R, R), H_cu), f_mul(2, f_mul(U1, H_sq)))
    Y3 = f_sub(f_mul(R, f_sub(f_mul(U1, H_sq), X3)), f_mul(S1, H_cu))
    Z3 = f_mul(f_mul(Z1, Z2), H)
    W3 = f_mul(Z3, Z3)

    return (X3, Y3, Z3, W3)