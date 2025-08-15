#if !defined(HELPER_CUH)
#define HELPER_CUH

// CUDA API CHECK MACRO
#define CUDA_CHECK(err)                                                        \
  {                                                                            \
    cudaError_t err_code = err;                                                \
    if (err_code != cudaSuccess) {                                             \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err_code) << " in "    \
                << __FILE__ << " at line " << __LINE__ << std::endl;           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#endif