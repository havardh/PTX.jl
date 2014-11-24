#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__attribute__((overloadable)) int getindex(__global int* A, long i) {
  return A[i];
}

__attribute__((overloadable)) int setindex(__global int* A, int v, long i) {
  return A[i] = v;
}

__attribute__((overloadable)) long getindex(__global long* A, long i) {
  return A[i];
}

__attribute__((overloadable)) long setindex(__global long* A, long v, long i) {
  return A[i] = v;
}

__attribute__((overloadable)) float getindex(__global float* A, long i) {
  return A[i];
}

__attribute__((overloadable)) float setindex(__global float* A, float v, long i) {
  return A[i] = v;
}

__attribute__((overloadable)) double getindex(__global double* A, long i) {
  return A[i];
}

__attribute__((overloadable)) double setindex(__global double* A, double v, long i) {
  return A[i] = v;
}