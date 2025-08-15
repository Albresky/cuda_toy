#include <cuda_runtime.h>
#include <iostream>

__global__ void add(float *x, float *y, float *z, int n) {
  // 获取全局索引
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  // 步长
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    z[i] = x[i] + y[i];
  }
}

int main() {
  int N = 1 << 20;
  int nBytes = N * sizeof(float);
  const int iterations = 100;

  // 申请host内存
  float *x, *y, *z;
  x = (float *)malloc(nBytes);
  y = (float *)malloc(nBytes);
  z = (float *)malloc(nBytes);

  // 初始化数据
  for (int i = 0; i < N; ++i) {
    x[i] = 10.0;
    y[i] = 20.0;
  }

  // 申请device内存
  float *d_x, *d_y, *d_z;
  cudaMalloc((void **)&d_x, nBytes);
  cudaMalloc((void **)&d_y, nBytes);
  cudaMalloc((void **)&d_z, nBytes);

  // 将host数据拷贝到device
  cudaMemcpy((void *)d_x, (void *)x, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_y, (void *)y, nBytes, cudaMemcpyHostToDevice);

  // 定义kernel的执行配置
  dim3 blockSize(512);
  dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

  // --- 性能分析 ---
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // 预热
  add<<<gridSize, blockSize>>>(d_x, d_y, d_z, N);

  // 计时开始
  cudaEventRecord(start);

  for (int i = 0; i < iterations; ++i) {
    add<<<gridSize, blockSize>>>(d_x, d_y, d_z, N);
  }

  // 计时结束并等待所有kernel执行完毕
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  std::cout << "迭代 " << iterations
            << " 次, 平均执行时间: " << milliseconds / iterations << " ms"
            << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  // --- 性能分析结束 ---

  // 将device得到的结果拷贝到host
  cudaMemcpy((void *)z, (void *)d_z, nBytes, cudaMemcpyDeviceToHost);

  // 检查执行结果
  float maxError = 0.0;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(z[i] - 30.0));
  std::cout << "最大误差: " << maxError << std::endl;

  // 释放device内存
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  // 释放host内存
  free(x);
  free(y);
  free(z);

  return 0;
}