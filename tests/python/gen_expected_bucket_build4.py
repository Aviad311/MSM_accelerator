from field import to_mont, ONE_M, INF
from jacobian import (
    jacobian_double,
    jacobian_mixed_add_mont,
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
# P3 = G again
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
ids = [1, 2, 3, 1]
points = [P0_aff, P1_aff, P2_aff, P3_aff]

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

print("Input affine Montgomery points:")
print()
print("P0 = G, bucket_id=1")
print(f"P0_X = 256'h{P0_aff[0]:064x};")
print(f"P0_Y = 256'h{P0_aff[1]:064x};")

print()
print("P1 = 2G affine, bucket_id=2")
print(f"P1_X = 256'h{P1_aff[0]:064x};")
print(f"P1_Y = 256'h{P1_aff[1]:064x};")

print()
print("P2 = 3G affine, bucket_id=3")
print(f"P2_X = 256'h{P2_aff[0]:064x};")
print(f"P2_Y = 256'h{P2_aff[1]:064x};")

print()
print("P3 = G, bucket_id=1")
print(f"P3_X = 256'h{P3_aff[0]:064x};")
print(f"P3_Y = 256'h{P3_aff[1]:064x};")

print()
print("Expected buckets, Jacobian Montgomery:")

print()
print("Bucket1 = G + G = 2G")
print(f"B1_X = 256'h{bucket1[0]:064x};")
print(f"B1_Y = 256'h{bucket1[1]:064x};")
print(f"B1_Z = 256'h{bucket1[2]:064x};")

print()
print("Bucket2 = 2G")
print(f"B2_X = 256'h{bucket2[0]:064x};")
print(f"B2_Y = 256'h{bucket2[1]:064x};")
print(f"B2_Z = 256'h{bucket2[2]:064x};")

print()
print("Bucket3 = 3G")
print(f"B3_X = 256'h{bucket3[0]:064x};")
print(f"B3_Y = 256'h{bucket3[1]:064x};")
print(f"B3_Z = 256'h{bucket3[2]:064x};")

print()
print("Affine check:")
print("bucket1 affine:", tuple(f"{v:064x}" for v in jacobian_to_affine(bucket1)))
print("bucket2 affine:", tuple(f"{v:064x}" for v in jacobian_to_affine(bucket2)))
print("bucket3 affine:", tuple(f"{v:064x}" for v in jacobian_to_affine(bucket3)))