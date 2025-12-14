from field import p
from jacobian import jacobian_to_affine
from msm_naive import msm_naive, is_on_curve
from msm_reference import msm_reference
from msm_pippenger import msm_pippenger


# ------------------------------------------------------------
# Base point of secp256k1
# ------------------------------------------------------------

Gx = 55066263022277343669578718895168534326250603453777594175500187360389116729240
Gy = 32670510020758816978083085130507043184471273380659243275938904335757337482424
G = (Gx, Gy)


# ------------------------------------------------------------
# Generate valid curve points for testing
# ------------------------------------------------------------

def generate_points_from_base(base_point, scalars):
    """
    Generate curve points as scalar multiples of base point.
    Uses naive scalar multiplication (affine).
    """
    points = []

    for k in scalars:
        P = msm_naive([k], [base_point])
        assert is_on_curve(P)
        points.append(P)

    return points


# ------------------------------------------------------------
# Main test driver
# ------------------------------------------------------------

def main():

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    w = 4   # small window for debugging
    scalars = [
        0x12345, 0x23456, 0x34567, 0x45678, 0x56789,
        0x6789A, 0x789AB, 0x89ABC, 0x9ABCD, 0xABCDE
    ]

    # --------------------------------------------------------
    # Generate curve points
    # --------------------------------------------------------

    print("[*] Generating curve points...")
    points = generate_points_from_base(G, list(range(1, len(scalars) + 1)))

    # --------------------------------------------------------
    # Naive MSM (Affine)
    # --------------------------------------------------------

    print("[*] Running naive MSM...")
    R_naive = msm_naive(scalars, points)
    assert is_on_curve(R_naive)

    # --------------------------------------------------------
    # Reference MSM (Window + Buckets)
    # --------------------------------------------------------

    print("[*] Running reference MSM...")
    R_ref_jacobian = msm_reference(scalars, points, w=w)
    R_ref = jacobian_to_affine(R_ref_jacobian)
    assert is_on_curve(R_ref)

    # --------------------------------------------------------
    # Compare results
    # --------------------------------------------------------

    print("\nResults:")
    print("Naive MSM:     ", R_naive)
    print("Reference MSM: ", R_ref)

    assert R_naive == R_ref
    print("\n✅ Reference MSM matches naive MSM")
    print("[*] Running Pippenger MSM...")
    R_fast_jacobian = msm_pippenger(scalars, points, w=w)
    R_fast = jacobian_to_affine(R_fast_jacobian)

    print("Pippenger MSM:", R_fast)

    assert R_fast == R_ref
    print("✅ Pippenger MSM matches reference MSM")


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    main()
