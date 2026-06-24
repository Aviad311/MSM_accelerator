from field import to_mont, from_mont, ONE_M
from jacobian import jacobian_double, jacobian_to_affine

Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

P = (
    to_mont(Gx),
    to_mont(Gy),
    ONE_M
)

D = jacobian_double(P)

print("Input Montgomery:")
print(f"X1 = 256'h{P[0]:064x};")
print(f"Y1 = 256'h{P[1]:064x};")
print(f"Z1 = 256'h{P[2]:064x};")

print()
print("Expected output Montgomery:")
print(f"EXP_X3 = 256'h{D[0]:064x};")
print(f"EXP_Y3 = 256'h{D[1]:064x};")
print(f"EXP_Z3 = 256'h{D[2]:064x};")

print()
print("Output affine normal:")
A = jacobian_to_affine(D)
print(f"x = {A[0]:064x}")
print(f"y = {A[1]:064x}")