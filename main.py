# ==========================================================
#   MSM Benchmark - Main (full, safe version)
# ==========================================================

import random
import op_counter

from op_counter import reset_counters

# MSM algorithms
from msm_naive import msm_naive
from msm_reference import msm_reference
from msm_pippenger import msm_pippenger
from msm_extended import msm_extended

# Affine conversions (NO counter versions)
from jacobian import jacobian_to_affine
from extended_jacobian import extended_to_affine

# Optional graph
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ----------------------------------------------------------
# Base point of secp256k1
# ----------------------------------------------------------
Gx = 55066263022277343669578718895168534326250603453777594175500187360389116729240
Gy = 32670510020758816978083085130507043184471273380659243275938904335757337482424
G = (Gx, Gy)


# ----------------------------------------------------------
# Parameters
# ----------------------------------------------------------
W = 8
N_LIST = [10, 50, 100]


# ----------------------------------------------------------
# Weighted cost model (ASIC-like)
# ----------------------------------------------------------
WEIGHTS = {
    "mul": 1.0,
    "add": 0.1,
    "sub": 0.1,
    "inv": 80.0,
}


def weighted_cost(c):
    return (
        c["mul"] * WEIGHTS["mul"] +
        c["add"] * WEIGHTS["add"] +
        c["sub"] * WEIGHTS["sub"] +
        c["inv"] * WEIGHTS["inv"]
    )


# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
def generate_random_scalars(num_scalars, bits=256):
    max_val = (1 << bits) - 1
    return [random.randint(1, max_val) for _ in range(num_scalars)]


def generate_points_from_base(base_point, scalars):
    points = []
    for k in scalars:
        P = msm_naive([k], [base_point])
        points.append(P)
    return points


def collect_field_counters():
    return {
        "mul": op_counter.field_mul_count,
        "add": op_counter.field_add_count,
        "sub": op_counter.field_sub_count,
        "inv": op_counter.field_inv_count,
    }


# ----------------------------------------------------------
# Plot
# ----------------------------------------------------------
def plot_weighted_cost(N_list, series):
    if not HAS_MPL:
        print("\nmatplotlib not installed – skipping graph.")
        return

    plt.figure()
    for name, y in series.items():
        plt.plot(N_list, y, marker="o", label=name)

    plt.xlabel("Number of scalars (N)")
    plt.ylabel("Weighted field-op cost")
    plt.title("MSM weighted cost vs N")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():

    # For graph
    series = {
        "Naive": [],
        "Reference": [],
        "Pippenger": [],
        "Extended": [],
    }

    for N in N_LIST:
        print(f"\n================ MSM test for N = {N} ================\n")

        scalars = generate_random_scalars(N, bits=256)
        points = generate_points_from_base(G, list(range(1, N + 1)))

        results_points = {}
        results_counts = {}

        # ---------------- Naive ----------------
        reset_counters()
        R_naive = msm_naive(scalars, points)
        results_points["Naive"] = R_naive
        naive_field = collect_field_counters()
        results_counts["Naive"] = {
            "aff": op_counter.affine_add_count,
            "jac": 0,
            "mix": 0,
            "dbl": 0,
            "ext_add": 0,
            "ext_mix": 0,
            "ext_dbl": 0,
            **naive_field,
        }

        # ---------------- Reference ----------------
        reset_counters()
        R_ref_jac = msm_reference(scalars, points, w=W)
        R_ref = jacobian_to_affine(R_ref_jac)
        results_points["Reference"] = R_ref
        ref_field = collect_field_counters()
        results_counts["Reference"] = {
            "aff": 0,
            "jac": op_counter.jacobian_add_count,
            "mix": op_counter.jacobian_mixed_add_count,
            "dbl": op_counter.jacobian_double_count,
            "ext_add": 0,
            "ext_mix": 0,
            "ext_dbl": 0,
            **ref_field,
        }

        # ---------------- Pippenger ----------------
        reset_counters()
        R_pip_jac = msm_pippenger(scalars, points, w=W)
        R_pip = jacobian_to_affine(R_pip_jac)
        results_points["Pippenger"] = R_pip
        pip_field = collect_field_counters()
        results_counts["Pippenger"] = {
            "aff": 0,
            "jac": op_counter.jacobian_add_count,
            "mix": op_counter.jacobian_mixed_add_count,
            "dbl": op_counter.jacobian_double_count,
            "ext_add": 0,
            "ext_mix": 0,
            "ext_dbl": 0,
            **pip_field,
        }

        # ---------------- Extended ----------------
        reset_counters()
        R_ext_ext = msm_extended(scalars, points, w=W)
        R_ext = extended_to_affine(R_ext_ext)
        results_points["Extended"] = R_ext
        ext_field = collect_field_counters()
        results_counts["Extended"] = {
            "aff": 0,
            "jac": 0,
            "mix": 0,
            "dbl": 0,
            "ext_add": op_counter.extended_add_count,
            "ext_mix": op_counter.extended_mixed_add_count,
            "ext_dbl": op_counter.extended_double_count,
            **ext_field,
        }

        # ---------------- Correctness ----------------
        assert results_points["Naive"] == results_points["Reference"] \
               == results_points["Pippenger"] == results_points["Extended"]

        print("✔ All MSM results match")
        print("MSM result (affine):")
        print(results_points["Naive"])

        # ---------------- Table ----------------
        print("\n================ Operation Count Comparison ================")
        print(
            "Model       | Aff add | Jac add | Mix add |  Dbl | "
            "Ext Add | Ext Mix | Ext Dbl |  F_Mul |  F_Add |  F_Sub |  F_Inv | WeightedCost"
        )
        print("-" * 150)

        for name, c in results_counts.items():
            wc = weighted_cost(c)
            print(
                f"{name:<11} | "
                f"{c['aff']:>7} | {c['jac']:>7} | {c['mix']:>7} | {c['dbl']:>5} | "
                f"{c['ext_add']:>7} | {c['ext_mix']:>7} | {c['ext_dbl']:>7} | "
                f"{c['mul']:>7} | {c['add']:>7} | {c['sub']:>7} | {c['inv']:>7} | "
                f"{wc:>11.0f}"
            )

        print("=" * 150)

        # ---------------- Collect for graph ----------------
        for algo in series:
            series[algo].append(weighted_cost(results_counts[algo]))

    # ---------------- Graph ----------------
    plot_weighted_cost(N_LIST, series)


# ----------------------------------------------------------
# Entry point
# ----------------------------------------------------------
if __name__ == "__main__":
    main()
