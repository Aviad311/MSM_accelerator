# ==========================================================
#   Field arithmetic for secp256k1  (GF(p))
#   Clean version for ASIC reference model
# ==========================================================
import op_counter

p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F

# ------------- Field operations -------------

def f_add(a, b):
    # Field addition
    op_counter.field_add_count += 1
    return (a + b) % p

def f_sub(a, b):
    # Field subtraction (counted separately)
    op_counter.field_sub_count += 1
    return (a - b) % p

def f_mul(a, b):
    # Field multiplication
    op_counter.field_mul_count += 1
    return (a * b) % p

def f_inv(a):
    # Field inversion (expensive)
    op_counter.field_inv_count += 1
    return pow(a, p - 2, p)

def f_neg(a):
    # Field negation (cheap, but counted for completeness)
    op_counter.field_neg_count += 1
    return (-a) % p

# Point at infinity (Jacobian representation)
INF = (1, 1, 0)
