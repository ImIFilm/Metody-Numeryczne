import numpy as np

tolerance = 10e-2

def read_nd_array_from_input(message="", dtype=float):
    n, m = map(int, input(message).split())
    a = np.zeros((n, m), dtype=dtype)
    for i in range(n):
        a[i] = [dtype(x) for x in input().split()]
    return a

def jacobi_solve(A, b):
    n = A.shape[0]
    b1 = np.transpose(b)[0]
    x = np.zeros(n)
    while True:
        x1 = np.zeros(n)
        for i in range(n):
            x1[i] = 1 / A[i][i] * (b1[i] - sum([A[i][j] * x[j] for j in range(n) if j != i]))
        error = sum(abs(np.dot(A, x1.transpose()) - b1))
        print(x1, x, error, np.dot(A, x1.transpose()) - b1)
        if error < tolerance:  
            return np.transpose(x1)
        else:
            x = x1

def main():
    A = read_nd_array_from_input("w pierwszej linii podaj rozmiar \"N N\" macierzy a potem podaj wiersze\n")
    b = read_nd_array_from_input("w pierwszej linii podaj rozmiar \"N 1\" macierzy a potem podaj wartosci\n")
    x = jacobi_solve(A, b)
    print("Szukany wektor x:", x, sep="\n")

if __name__ == "__main__":
    main()
