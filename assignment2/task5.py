import numpy as np
import time
A = np.array([
    [5, 1, 1],
    [1, 4, 2],
    [1, 1, 5]
])
b = np.array([10, 12, 15])
def relaxation_method(A, b, omega, tol=1e-6, max_iterations=1000):
    n = len(b)
    x = np.zeros(n)
    iterations = 0

    for _ in range(max_iterations):
        x_new = np.copy(x)
        for i in range(n):
            #Represents the sum of terms in row i, excluding the diagonal term
            sigma = sum(A[i][j] * x_new[j] for j in range(n) if j != i)
            x_new[i] = (1 - omega) * x[i] + (omega / A[i][i]) * (b[i] - sigma) #Formula
        iterations += 1
        if np.linalg.norm(x_new - x, ord=np.inf) < tol: #Convergence Check
            return x_new, iterations
        x = x_new
    raise ValueError("Solution did not converge within the maximum number of iterations")

omega1 = 1.1 # Relaxation parameter
start_time_omega1 = time.time()
solution_omega1, iterations_omega1 = relaxation_method(A, b, omega1)
time_omega1 = time.time() - start_time_omega1

omega2 = 1.5 # Relaxation parameter
start_time_omega2 = time.time()
solution_omega2, iterations_omega2 = relaxation_method(A, b, omega2)
time_omega2 = time.time() - start_time_omega2

print("Results with omega = 1.1:")
print("Solution:", solution_omega1)
print("Iterations:", iterations_omega1)
print("Execution Time:", time_omega1, "seconds")

print("\nResults with omega = 1.5:")
print("Solution:", solution_omega2)
print("Iterations:", iterations_omega2)
print("Execution Time:", time_omega2, "seconds")

