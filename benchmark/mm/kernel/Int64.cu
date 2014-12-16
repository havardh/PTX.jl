

extern "C"
{
  __device__ long computeCell(const long *A, const long *B, long *C,
			      int row, int col, long n) {
    long v=0;
    if (row < n && col < n) {
      for (long i=0; i<n; i++) {
	v += A[i + row*n] * B[col + i*n];
      }
    }
    return v;
  }
  
  __global__ void MatrixMultiply(
    const long *A, const long *B, long *C, long n) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    C[col + row*n] = computeCell(A, B, C, row, col, n);
  }
}