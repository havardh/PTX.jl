extern "C"
{
  
  __device__ float computeCell(const float *A, const float *B, float *C, int row, int col, long n) {
    
    float v=0;
    if (row < n && col < n) {
      for (long i=0; i<n; i++) {
	v += A[i + row*n] * B[col + i*n];
      }
    }
    return v;
  }
  
  __global__ void MatrixMultiply(const float *A, const float *B, float *C, long n) {
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    C[col + row*n] = computeCell(A, B, C, row, col, n);
  }
  
}