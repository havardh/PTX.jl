long getindex(__global long* A, long i) {
  return A[i];
}

long setindex(__global long* A, long v, long i) {
  return A[i] = v;
}
