import time
import matplotlib.pyplot as plt
import numpy as np
from numba import cuda
from tabulate import tabulate

TPB = 32
ITER = 100


@cuda.jit
def gpu_vec_sum(vec, res):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    idx = tx + bx * TPB

    if idx < vec.shape[0]:
        cuda.atomic.add(res, 0, vec[idx])


def cpu_vec_sum(vec):
    return np.sum(vec)


def calculation():
    rows = []
    vec_sizes = np.linspace(1000, 10000000, 5, dtype=int)

    for vec_size in vec_sizes:
        gpu_time = 0
        cpu_time = 0

        for _ in range(ITER):
            vec = np.ones(vec_size)
            res = np.zeros(1, dtype=np.int32)
            g_vec = cuda.to_device(vec)
            g_res = cuda.to_device(res)

            start = time.time()
            gpu_vec_sum[int((vec_size + TPB) / TPB), TPB](g_vec, g_res)
            gpu_time += time.time() - start

            gpu_res = g_res.copy_to_host()

            start = time.time()
            cpu_res = cpu_vec_sum(vec)
            cpu_time += time.time() - start

        row = [vec_size, cpu_res, cpu_time / ITER, gpu_res, gpu_time / ITER]
        rows.append(row)

    print(tabulate(rows, headers=['vector size', 'cpu sum', 'cpu, sec', 'gpu sum', 'gpu, sec']))
    return rows


output_data = calculation()

vec_array = list(map(lambda x: x[0], output_data))
cpu_time_array = list(map(lambda x: x[2], output_data))
gpu_time_array = list(map(lambda x: x[4], output_data))
acceleration_array = list(map(lambda x: x[2] / x[4], output_data))

plt.plot(vec_array, cpu_time_array, label='CPU', color='red', linestyle='-')
plt.plot(vec_array, gpu_time_array, label='GPU', color='green', linestyle='-')
plt.title('CPU и GPU')
plt.xlabel('Размер')
plt.ylabel('Время, с')
plt.legend()

plt.figure()
plt.title("Ускорение")
plt.plot(vec_array, acceleration_array)
plt.xlabel("Размер")
plt.grid()

plt.show()
