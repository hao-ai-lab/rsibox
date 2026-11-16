// a native kernel source: counted by file
__global__ void norm(float* x, int n) {
  for (int i = 0; i < n; ++i) x[i] *= 2;
}
