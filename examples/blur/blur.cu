#define N 256

extern "C"
{

  __global__ void blur(const float *IN,
		       float *OUT)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    float v = 0;
    if (!(idx==0 || idx==N-1 || idy == 0 || idy==N-1) ) {
      for(int i=-1; i<2; i++) {
	for (int j=-1; j<2; j++) {
	  v = v + IN[(idx+i) + (idy+j)*N];
	}
      }
      v = v / 9;
    } else {
      v = IN[idx + idy*N];
    }
    
    OUT[idx + idy*N] = v;
  }
  
}