import math
import numpy as np
from PIL import Image
import time
from numba import cuda
from matplotlib import pyplot as plt

@cuda.jit
def gpu_kernel(img, result):
    i, j = cuda.grid(2)
    if i < result.shape[0] and j < result.shape[1]:
        x = i // 2
        y = j // 2
        if x < img.shape[0] - 1 and y < img.shape[1] - 1:
            fx = i % 2
            fy = j % 2
            fx1 = min(x + 1, img.shape[0] - 1)
            fy1 = min(y + 1, img.shape[1] - 1)
            result[i, j] = (img[x, y] * (1 - fx) * (1 - fy) +
                            img[x, fy1] * (1 - fx) * fy +
                            img[fx1, y] * fx * (1 - fy) +
                            img[fx1, fy1] * fx * fy)

def gpu_bilinear(img):
    threadsperblock = (8, 8)
    blockspergrid = (math.ceil(img.shape[0] * 2 / threadsperblock[0]),
                     math.ceil(img.shape[1] * 2 / threadsperblock[1]))
    img_arr_gpu = cuda.to_device(img)
    result_gpu = cuda.to_device(np.zeros((img.shape[0] * 2, img.shape[1] * 2), dtype=np.uint8))
    start = time.perf_counter()
    gpu_kernel[blockspergrid, threadsperblock](img_arr_gpu, result_gpu)
    end = time.perf_counter()
    result = result_gpu.copy_to_host()
    img_gpu = Image.fromarray(result.reshape((img.shape[0] * 2, img.shape[1] * 2)))
    img_gpu = img_gpu.convert("L")
    img_gpu.save('bilinear_gpu_{}x{}.bmp'.format(img.shape[1], img.shape[0]))
    return end - start

def cpu_bilinear(img):
    result = np.zeros((img.shape[0] * 2, img.shape[1] * 2), dtype=np.float32)
    start = time.perf_counter()
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            x = i // 2
            y = j // 2
            if x < img.shape[0] - 1 and y < img.shape[1] - 1:
                fx = i % 2
                fy = j % 2
                fx1 = min(x + 1, img.shape[0] - 1)
                fy1 = min(y + 1, img.shape[1] - 1)
                result[i, j] = (img[x, y] * (1 - fx) * (1 - fy) +
                                img[x, fy1] * (1 - fx) * fy +
                                img[fx1, y] * fx * (1 - fy) +
                                img[fx1, fy1] * fx * fy)
    end = time.perf_counter()
    img_cpu = Image.fromarray(result.reshape((img.shape[0] * 2, img.shape[1] * 2)))
    img_cpu = img_cpu.convert("L")
    img_cpu.save('bilinear_cpu_{}x{}.bmp'.format(img.shape[1], img.shape[0]))
    return end - start

if __name__ == '__main__':
    images = ["100.bmp", "200.bmp", "400.bmp", "600.bmp"]

    mas_size = [100, 200, 400, 600]
    array_time_CPU = []
    array_time_GPU = []
    acel = []

    for image in images:
        img_arr = np.array(Image.open(image).convert("L"))
        res_dict = {"resolution inp": (img_arr.shape[1], img_arr.shape[0])}
        array_time_CPU.append(cpu_bilinear(img_arr))
        array_time_GPU.append(gpu_bilinear(img_arr))

    plt.plot(mas_size, array_time_CPU, label='CPU', color='red', linestyle='-')
    plt.plot(mas_size, array_time_GPU, label='GPU', color='green', linestyle='-')
    plt.title('CPU и GPU')
    plt.xlabel('Размер')
    plt.ylabel('Время, с')
    plt.legend()
    plt.show()

    acel = []
    for i in range(len(array_time_CPU)):
        acel.append(array_time_CPU[i] / array_time_GPU[i])
    plt.plot(mas_size, acel, label='Ускорение')
    plt.title('Ускорение')
    plt.xlabel('Размер')
    plt.legend()
    plt.show()
