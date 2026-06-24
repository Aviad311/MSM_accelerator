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

# Build affine points:
# P0 = G
# P1 = 2G
# P2 = 3G
# P3 = G
P0_aff = G_aff

P1_jac = jacobian_double(G_jac)
P1_aff_normal = jacobian_to_affine(P1_jac)
P1_aff = (
    to_mont(P1_aff_normal[0]),
    to_mont(P1_aff_normal[1]),
)

P2_jac = jacobian_mixed_add_mont(P1_jac, G_aff)
P2_aff_normal = jacobian_to_affine(P2_jac)
P2_aff = (
    to_mont(P2_aff_normal[0]),
    to_mont(P2_aff_normal[1]),
)

P3_aff = G_aff

# bucket ids:
# P0 -> bucket1
# P1 -> bucket2
# P2 -> bucket3
# P3 -> bucket1
points = [P0_aff, P1_aff, P2_aff, P3_aff]
ids    = [1,      2,      3,      1]

# ------------------------------------------------------------
# Bucket build
# ------------------------------------------------------------
bucket1 = INF
bucket2 = INF
bucket3 = INF

for point, bid in zip(points, ids):
    if bid == 1:
        bucket1 = jacobian_mixed_add_mont(bucket1, point)
    elif bid == 2:
        bucket2 = jacobian_mixed_add_mont(bucket2, point)
    elif bid == 3:
        bucket3 = jacobian_mixed_add_mont(bucket3, point)

# ------------------------------------------------------------
# Bucket reduction
#
# running_sum = INF
# result      = INF
#
# for bucket in [bucket3, bucket2, bucket1]:
#     running_sum = running_sum + bucket
#     result      = result + running_sum
# ------------------------------------------------------------
running_sum = INF
result = INF

for B in [bucket3, bucket2, bucket1]:
    running_sum = jacobian_add(running_sum, B)
    result = jacobian_add(result, running_sum)

print("Input affine Montgomery points and bucket ids:")
print()
print("P0 = G, bucket_id=1")
print(f"P0_X = 256'h{P0_aff[0]:064x};")
print(f"P0_Y = 256'h{P0_aff[1]:064x};")
print("P0_bid = 2'd1;")

print()
print("P1 = 2G affine, bucket_id=2")
print(f"P1_X = 256'h{P1_aff[0]:064x};")
print(f"P1_Y = 256'h{P1_aff[1]:064x};")
print("P1_bid = 2'd2;")

print()
print("P2 = 3G affine, bucket_id=3")
print(f"P2_X = 256'h{P2_aff[0]:064x};")
print(f"P2_Y = 256'h{P2_aff[1]:064x};")
print("P2_bid = 2'd3;")

print()
print("P3 = G, bucket_id=1")
print(f"P3_X = 256'h{P3_aff[0]:064x};")
print(f"P3_Y = 256'h{P3_aff[1]:064x};")
print("P3_bid = 2'd1;")

print()
print("Intermediate buckets after bucket_build:")
print("Bucket1:")
print(f"B1_X = 256'h{bucket1[0]:064x};")
print(f"B1_Y = 256'h{bucket1[1]:064x};")
print(f"B1_Z = 256'h{bucket1[2]:064x};")

print()
print("Bucket2:")
print(f"B2_X = 256'h{bucket2[0]:064x};")
print(f"B2_Y = 256'h{bucket2[1]:064x};")
print(f"B2_Z = 256'h{bucket2[2]:064x};")

print()
print("Bucket3:")
print(f"B3_X = 256'h{bucket3[0]:064x};")
print(f"B3_Y = 256'h{bucket3[1]:064x};")
print(f"B3_Z = 256'h{bucket3[2]:064x};")

print()
print("Expected window result after reduce:")
print(f"EXP_X = 256'h{result[0]:064x};")
print(f"EXP_Y = 256'h{result[1]:064x};")
print(f"EXP_Z = 256'h{result[2]:064x};")

print()
print("Expected affine normal:")
A = jacobian_to_affine(result)
print(f"x = {A[0]:064x}")
print(f"y = {A[1]:064x}")