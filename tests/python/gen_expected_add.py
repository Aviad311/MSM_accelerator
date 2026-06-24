from field import to_mont, ONE_M
from jacobian import (
    jacobian_double,
    jacobian_mixed_add_mont,
    jacobian_add,
    jacobian_to_affine,
)

Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

G_aff_mont = (
    to_mont(Gx),
    to_mont(Gy)
)

G_jac_mont = (
    to_mont(Gx),
    to_mont(Gy),
    ONE_M
)

# P = 2G
P = jacobian_double(G_jac_mont)

# Q = 3G = 2G + G
Q = jacobian_mixed_add_mont(P, G_aff_mont)

# R = 2G + 3G = 5G
R = jacobian_add(P, Q)

print("Input P = 2G Jacobian Montgomery:")
print(f"P_X1 = 256'h{P[0]:064x};")
print(f"P_Y1 = 256'h{P[1]:064x};")
print(f"P_Z1 = 256'h{P[2]:064x};")

print()
print("Input Q = 3G Jacobian Montgomery:")
print(f"Q_X2 = 256'h{Q[0]:064x};")
print(f"Q_Y2 = 256'h{Q[1]:064x};")
print(f"Q_Z2 = 256'h{Q[2]:064x};")

print()
print("Expected R = 5G Jacobian Montgomery:")
print(f"EXP_X3 = 256'h{R[0]:064x};")
print(f"EXP_Y3 = 256'h{R[1]:064x};")
print(f"EXP_Z3 = 256'h{R[2]:064x};")

print()
print("Expected R affine normal:")
A = jacobian_to_affine(R)
print(f"x = {A[0]:064x}")
print(f"y = {A[1]:064x}")