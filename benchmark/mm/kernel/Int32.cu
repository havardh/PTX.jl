

extern "C"
{
  __device__ int computeCell(const int *A, const int *B, int *C,
			      int row, int col, long n) {
    int v=0;
    if (row < n && col < n) {
      for (int i=0; i<n; i++) {
	v += A[i + row*n] * B[col + i*n];
      }
    }
    return v;
  }
  
  __global__ void MatrixMultiply(
    const int *A, const int *B, int *C, long n) {

    int col = (blockIdx.x * blockDim.x + threadIdx.x);
    int row = (blockIdx.y * blockDim.y + threadIdx.y);

    C[col + row*n] = computeCell(A, B, C, row, col, n);
  }
  
}