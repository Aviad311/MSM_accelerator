from field import f_mul, p, INF
import op_counter
# Point in Jacobian-4: (X, Y, Z, Z2)
# where Z2 = Z^2 mod p
def affine_to_jacobian4(Q):
    """
    Convert affine point Q = (x, y)
    to Jacobian-4 point (X, Y, Z, Z2)
    """
    if Q is None:
        return (0, 1, 0, 0)  # INF

    x, y = Q
    return (x % p, y % p, 1, 1)  # Z = 1, Z2 = 1
