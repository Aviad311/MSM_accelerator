from field import to_mont, ONE_M, INF
from jacobian import (
    jacobian_double,
    jacobian_mixed_add_mont,
    jacobian_add,
    jacobian_to_affine,
)

Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

G_aff = (
    to_mont(Gx),
    to_mont(Gy),
)

G_jac = (
    to_mont(Gx),
    to_mont(Gy),
    ONE_M,
)

# Build points:
# B1 = G
# B2 = 2G
# B3 = 3G
B1 = G_jac
B2 = jacobian_double(G_jac)
B3 = jacobian_mixed_add_mont(B2, G_aff)

# Reduction for buckets [1..3]:
#
# running_sum = INF
# result = INF
#
# for i = 3 downto 1:
#     running_sum = running_sum + bucket[i]
#     result      = result + running_sum
#
running_sum = INF
result = INF

for B in [B3, B2, B1]:
    running_sum = jacobian_add(running_sum, B)
    result = jacobian_add(result, running_sum)

print("Bucket inputs, Jacobian Montgomery:")
print()
print("B1 = G:")
print(f"B1_X = 256'h{B1[0]:064x};")
print(f"B1_Y = 256'h{B1[1]:064x};")
print(f"B1_Z = 256'h{B1[2]:064x};")

print()
print("B2 = 2G:")
print(f"B2_X = 256'h{B2[0]:064x};")
print(f"B2_Y = 256'h{B2[1]:064x};")
print(f"B2_Z = 256'h{B2[2]:064x};")

print()
print("B3 = 3G:")
print(f"B3_X = 256'h{B3[0]:064x};")
print(f"B3_Y = 256'h{B3[1]:064x};")
print(f"B3_Z = 256'h{B3[2]:064x};")

print()
print("Expected reduce result:")
print(f"EXP_X = 256'h{result[0]:064x};")
print(f"EXP_Y = 256'h{result[1]:064x};")
print(f"EXP_Z = 256'h{result[2]:064x};")

print()
print("Expected affine normal:")
A = jacobian_to_affine(result)
print(f"x = {A[0]:064x}")
print(f"y = {A[1]:064x}")