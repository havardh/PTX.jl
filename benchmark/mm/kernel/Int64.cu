extern "C"
{

  __global__ void MatrixMultiply(
    const long *A, const int *B, int *C,
    int n, int m, int k) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    long v=0;
    for (int i=0; i<m; i++) {
      v += A[i + row*m] * B[col + i*k];
    }

    C[col + row*k] = v;
  }
  
}