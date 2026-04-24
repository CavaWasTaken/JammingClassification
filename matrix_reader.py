# I want to read matrix in numpy format
import numpy as np
def read_matrix(file_path):
    """
    Reads a matrix from a file in numpy format.

    Parameters:
    file_path (str): The path to the file containing the matrix.

    Returns:
    np.ndarray: The matrix read from the file.
    """
    try:
        matrix = np.load(file_path)
        return matrix
    except Exception as e:
        print(f"An error occurred while reading the matrix: {e}")
        return None
# Example usage:
matrix = read_matrix("C:/Users/loren/Downloads/6.npy")
# salva la matrice in csv
np.savetxt("C:/Users/loren/Downloads/6.csv", matrix, delimiter=",")
print(matrix)
