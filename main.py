from msm import msm

def main():
    scalars = [123456, 654321, 11111]
    points = [(1,2), (3,4), (5,6)]  # רק דוגמה

    R = msm(scalars, points, w=16)
    print("Result =", R)

if __name__ == "__main__":
    main()
