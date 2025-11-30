import numpy as np

# Define the matrix A and vector b
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 10]])

b = np.array([3, 3, 4])

# Solve the system using numpy.linalg.solve
x_standard = np.linalg.solve(A, b)

# Output the solution
print("Solution using numpy.linalg.solve:", x_standard)
