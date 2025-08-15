#if !defined(GEMM_CUH)
#define GEMM_CUH

/////////////////////////////////
// Optimize Levels (Ascending) //
/////////////////////////////////
/* Navie implementation, elem-by-elem */
// #define OP0

/* -- Based on L0 , using tiling, elem-by-elem though -- */
// #define OP1

/* -- Based on L1, using tiling, 1-thread --> region though -- */
// #define OP2

/* -- Based on L2, using sync ping-pong buffer -- */
// #define OP3

/* -- Based on L2, using async ping-pong buffer (real pipeline) -- */
// #define OP4

/* -- Based on L2, unrolling loop -- */
// #define OP5

/* Based on L2, using Tensor Cores */
#define OP6

#define IDX2C(r, c, ld) ((r) * (ld) + (c))

#define ITER 100

struct Matrix {
  int M;
  int K;
  int N;
  float *elements;

public:
  Matrix(int m = 0, int k = 0, int n = 0)
      : M(m), K(k), N(n), elements(nullptr) {}
};
#endif