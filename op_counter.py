# ==========================================================
# Global operation counters
# ==========================================================

# ---- Point operation counters ----
affine_add_count = 0

jacobian_add_count = 0
jacobian_mixed_add_count = 0
jacobian_double_count = 0

extended_add_count = 0
extended_mixed_add_count = 0
extended_double_count = 0

# ---- Field operation counters ----
field_add_count = 0
field_sub_count = 0
field_mul_count = 0
field_inv_count = 0
field_neg_count = 0


def reset_counters():
    global affine_add_count
    global jacobian_add_count, jacobian_mixed_add_count, jacobian_double_count
    global extended_add_count, extended_mixed_add_count, extended_double_count
    global field_add_count, field_sub_count, field_mul_count, field_inv_count, field_neg_count

    # Point ops
    affine_add_count = 0

    jacobian_add_count = 0
    jacobian_mixed_add_count = 0
    jacobian_double_count = 0

    extended_add_count = 0
    extended_mixed_add_count = 0
    extended_double_count = 0

    # Field ops
    field_add_count = 0
    field_sub_count = 0
    field_mul_count = 0
    field_inv_count = 0
    field_neg_count = 0


def print_counters(title="Operation counts"):
    print(f"\n--- {title} ---")

    # Point ops
    print("Affine add          :", affine_add_count)
    print("Jacobian add        :", jacobian_add_count)
    print("Jacobian mixed add  :", jacobian_mixed_add_count)
    print("Jacobian double     :", jacobian_double_count)

    print("Extended add        :", extended_add_count)
    print("Extended mixed add  :", extended_mixed_add_count)
    print("Extended double     :", extended_double_count)

    # Field ops
    print("Field adds          :", field_add_count)
    print("Field subs          :", field_sub_count)
    print("Field muls          :", field_mul_count)
    print("Field invs          :", field_inv_count)
    print("Field negs          :", field_neg_count)

    total_field_ops = field_add_count + field_sub_count + field_mul_count + field_inv_count + field_neg_count
    print("Total field ops     :", total_field_ops)

