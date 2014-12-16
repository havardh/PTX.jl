extern "C"
{

  __global__ void blur(const long *IN, long *OUT, const int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    long v = 0;
    if (!(idx==0 || idx==n-1 || idy == 0 || idy==n-1) ) {
      for(int i=-1; i<2; i++) {
	for (int j=-1; j<2; j++) {
	  v = v + IN[(idx+i) + (idy+j)*n];
	}
      }
      v = v / 9;
    } else {
      v = IN[idx + idy*n];
    }
    
    OUT[idx + idy*n] = v;
  }
  
}