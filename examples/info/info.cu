/*
 * @Author: Albresky albre02@outlook.com
 * @Date: 2025-08-15 10:41:02
 * @LastEditors: Albresky albre02@outlook.com
 * @LastEditTime: 2025-08-15 17:13:11
 * @FilePath: /cuda/examples/info.cu
 * @Description: Dump NVida device
 */
#include <cuda_runtime.h>
#include <iostream>

#define CHECK(call)                                                            \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "     \
                << cudaGetErrorString(err) << std::endl;                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

int main() {
  cudaDeviceProp devProp;
  for (int dev = 0; dev < 4; ++dev) {
    CHECK(cudaGetDeviceProperties(&devProp, dev));
    std::cout << "\n#########################################\n";
    std::cout << "使用 GPU device " << dev << ": " << devProp.name << std::endl;
    std::cout << "SM 数量：" << devProp.multiProcessorCount << std::endl;
    std::cout << "每个线程块的共享内存大小："
              << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock
              << std::endl;
    std::cout << "每个 EM 的最大线程数：" << devProp.maxThreadsPerMultiProcessor
              << std::endl;
    std::cout << "每个 SM 的最大线程束数："
              << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
  }
  return 0;
}