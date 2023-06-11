/*
 * File: kernel.cu
 * Project: cuda_src
 * Created Date: 11/06/2023
 * Author: Shun Suzuki
 * -----
 * Last Modified: 11/06/2023
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2023 Shun Suzuki. All rights reserved.
 *
 */

#include <cstdint>

__global__ void add_kernel(const double *x, double *y, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    y[i] += x[i];
  }
}

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

void cu_add(const double *x, double *y, const int32_t len) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (len + threadsPerBlock - 1) / threadsPerBlock;
  add_kernel<<<blocksPerGrid, threadsPerBlock>>>(x, y, len);
}

#ifdef __cplusplus
}
#endif
