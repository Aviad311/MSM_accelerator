# ==========================================================
#   Field arithmetic for secp256k1  (GF(p))
#   Clean version for ASIC reference model
# ==========================================================

p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F

# ------------- Field operations -------------

def f_add(a, b):
    return (a + b) % p

def f_sub(a, b):
    return (a - b) % p

def f_mul(a, b):
    return (a * b) % p

def f_inv(a):
    return pow(a, p - 2, p)

def f_neg(a):
    return (-a) % p

# Point at infinity can also live here if you prefer:
INF = (1, 1, 0)
