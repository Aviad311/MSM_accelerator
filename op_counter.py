# Global operation counters

jacobian_add_count = 0
jacobian_mixed_add_count = 0
jacobian_double_count = 0
affine_add_count = 0

jacobian_add_count = 0
extended_double_count = 0

extended_mixed_add_count = 0
extended_add_count = 0
extended_double_count = 0

field_mul_count = 0
def reset_counters():
    global jacobian_add_count, jacobian_mixed_add_count, jacobian_double_count, affine_add_count
    global jacobian_double_count, affine_add_count
    global extended_mixed_add_count, extended_add_count, extended_double_count

    global field_mul_count

    jacobian_add_count = 0
    jacobian_mixed_add_count = 0
    jacobian_double_count = 0
    affine_add_count = 0
    extended_double_count = 0
    field_mul_count = 0
    extended_mixed_add_count = 0
    extended_add_count = 0
    extended_double_count = 0


def print_counters(title="Operation counts"):
    print(f"\n--- {title} ---")
    print("Jacobian add        :", jacobian_add_count)
    print("Jacobian mixed add  :", jacobian_mixed_add_count)
    print("Jacobian double     :", jacobian_double_count)
    print("Affine add          :", affine_add_count)

    print("Extended mixed add  :", extended_mixed_add_count)
    print("Extended add        :", extended_add_count)
    print("Extended double     :", extended_double_count)

    print("Extended double     :", extended_double_count)
    print("Total Field Muls    :", field_mul_count)
