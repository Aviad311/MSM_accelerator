from field import p
from jacobian import jacobian_to_affine
from msm_naive import msm_naive, is_on_curve
from msm_reference import msm_reference
from msm_pippenger import msm_pippenger
from op_counter import reset_counters, print_counters
import op_counter

from extended_jacobian import extended_to_affine
from msm_extended import msm_extended

import random

def generate_random_scalars(num_scalars, bits=32):
    max_val = (1 << bits) - 1
    return [random.randint(1, max_val) for _ in range(num_scalars)]

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
# Pretty print comparison table
# ------------------------------------------------------------

def print_comparison_table():
    print("\n================ Operation Count Comparison ================")
    # הוספתי עמודה בסוף: Total Muls
    print(
        f"{'Model':<15} | {'Aff add':>7} | {'Jac add':>7} | {'Mix add':>7} | {'Dbl':>7} | {'Ext Add':>7} | {'Ext Mix':>7} | {'Ext Dbl':>7} | {'Total Muls':>11}")
    print("-" * 118)

    models = [
        ("Naive", naive_counts),
        ("Reference", reference_counts),
        ("Pippenger", pippenger_counts),
        ("Extended", extended_counts),
    ]

    for name, c in models:
        print(
            f"{name:<15} | "
            f"{c.get('affine', 0):>7} | "
            f"{c.get('jac', 0):>7} | "
            f"{c.get('mixed', 0):>7} | "
            f"{c.get('double', 0):>7} | "
            f"{c.get('ext_add', 0):>7} | "
            f"{c.get('ext_mixed', 0):>7} | "
            f"{c.get('ext_double', 0):>7} | "
            f"{c.get('total_mul', 0):>11}"  # <--- שליפת הנתון החדש
        )
    print("=" * 118)


# ------------------------------------------------------------
# Main test driver
# ------------------------------------------------------------

def main():

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    w = 8  # window size (small for debugging)
    scalars = generate_random_scalars(num_scalars=2000, bits=256)

    for s in scalars:
        print(hex(s))

    # --------------------------------------------------------
    # Generate curve points
    # --------------------------------------------------------

    print("[*] Generating curve points...")
    points = generate_points_from_base(G, list(range(1, len(scalars) + 1)))

    # --------------------------------------------------------
    # Naive MSM (Affine)
    # --------------------------------------------------------

    print("\n[*] Running Naive MSM...")
    reset_counters()
    R_naive = msm_naive(scalars, points)
    assert is_on_curve(R_naive)
    print("Naive MSM result:", R_naive)
    print_counters("Naive MSM")

    global naive_counts
    naive_counts = {
        "affine": op_counter.affine_add_count,
        "jac": op_counter.jacobian_add_count,
        "mixed": op_counter.jacobian_mixed_add_count,
        "double": op_counter.jacobian_double_count,
        "total_mul": op_counter.field_mul_count
    }

    # --------------------------------------------------------
    # Reference MSM (Window + Buckets)
    # --------------------------------------------------------

    print("\n[*] Running Reference MSM...")
    reset_counters()
    R_ref_jacobian = msm_reference(scalars, points, w=w)
    R_ref = jacobian_to_affine(R_ref_jacobian)
    assert is_on_curve(R_ref)
    print("Reference MSM result:", R_ref)
    print_counters("Reference MSM")

    global reference_counts
    reference_counts = {
        "affine": op_counter.affine_add_count,
        "jac": op_counter.jacobian_add_count,
        "mixed": op_counter.jacobian_mixed_add_count,
        "double": op_counter.jacobian_double_count,
        "total_mul": op_counter.field_mul_count
    }

    # --------------------------------------------------------
    # Pippenger MSM (Fast)
    # --------------------------------------------------------

    print("\n[*] Running Pippenger MSM...")
    reset_counters()
    R_fast_jacobian = msm_pippenger(scalars, points, w=w)
    R_fast = jacobian_to_affine(R_fast_jacobian)
    assert is_on_curve(R_fast)
    print("Pippenger MSM result:", R_fast)
    print_counters("Pippenger MSM")

    global pippenger_counts
    pippenger_counts = {
        "affine": op_counter.affine_add_count,
        "jac": op_counter.jacobian_add_count,
        "mixed": op_counter.jacobian_mixed_add_count,
        "double": op_counter.jacobian_double_count,
        "total_mul": op_counter.field_mul_count
    }

    # --------------------------------------------------------
    # Extended MSM (XYZW)
    # --------------------------------------------------------

    print("\n[*] Running Extended MSM (XYZW)...")
    reset_counters()

    # הרצת ה-MSM
    R_ext_extended = msm_extended(scalars, points, w=w)

    # המרה חזרה לאפיני כדי שנוכל להשוות
    R_ext = extended_to_affine(R_ext_extended)

    assert is_on_curve(R_ext)
    print("Extended MSM result:", R_ext)
    print_counters("Extended MSM")

    global extended_counts
    extended_counts = {
        "affine": op_counter.affine_add_count,  # אמור להיות 0
        "jac": op_counter.jacobian_add_count,  # אמור להיות 0
        "mixed": op_counter.jacobian_mixed_add_count,  # אמור להיות 0
        "double": op_counter.jacobian_double_count,  # אמור להיות 0
        # המונים החדשים:
        "ext_add": op_counter.extended_add_count,
        "ext_mixed": op_counter.extended_mixed_add_count,
        "ext_double": op_counter.extended_double_count,
        "total_mul": op_counter.field_mul_count
    }

    # --------------------------------------------------------
    # Final correctness check
    # --------------------------------------------------------

    # עדכון שורת הבדיקה
    assert R_naive == R_ref == R_fast == R_ext
    print("\n✅ All MSM results match!")

    # --------------------------------------------------------
    # Comparison table
    # --------------------------------------------------------

    print_comparison_table()


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    main()
