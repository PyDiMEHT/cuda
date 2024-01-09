import numpy as np
import time
from numba import cuda
import math
from matplotlib import pyplot as plt

matrix_size = 100

# Initialization of matrices for CPU
cpu_matrix1 = np.ones((matrix_size, matrix_size), dtype=int)
cpu_matrix2 = np.ones((matrix_size, matrix_size), dtype=int)
cpu_matrix_res = np.zeros((matrix_size, matrix_size), dtype=int)

# Initialization of matrices for GPU
gpu_matrix1 = cuda.to_device(cpu_matrix1)
gpu_matrix2 = cuda.to_device(cpu_matrix2)
gpu_matrix_res = cuda.to_device(cpu_matrix_res)


def cpu_mat_mul(A, B, C):
    for i in range(matrix_size):
        for j in range(matrix_size):
            res = 0
            for k in range(matrix_size):
                res += A[i, k] * B[k, j]
            C[i, j] = res


@cuda.jit
def gpu_mat_mul(A, B, C):
    for i in range(matrix_size):
        for j in range(matrix_size):
            rez = 0
            for z in range(matrix_size):
                rez += A[i, z] * B[z, j]
            C[i, j] = rez


def cpu_calc():
    print("CPU begin working.")
    start_time = time.time()
    cpu_mat_mul(cpu_matrix1, cpu_matrix2, cpu_matrix_res)
    print("%s CPU" % (time.time() - start_time))


def gpu_calc():
    # Kernel parameters
    threadsperblock = (32, 32)
    blockspergrid_x = int(math.ceil(cpu_matrix1.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(cpu_matrix2.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    start_time = time.time()
    gpu_mat_mul[blockspergrid, threadsperblock](gpu_matrix1, gpu_matrix2, gpu_matrix_res)
    print("%s GPU" % (time.time() - start_time))


if __name__ == "__main__":
    cpu_calc()
    gpu_calc()
    result_GPU = gpu_matrix_res.copy_to_host()
    print(f" Проверка равенства матриц {np.array_equal(cpu_matrix_res, result_GPU)}")
