module OpenCLBlur

using BenchmarkLite

import OpenCL
const cl = OpenCL

const kernel = "
#define N 4
#define M 4

__kernel void blur(__global const long *IN,
                   __global long *OUT)
{
  int idx = get_global_id(0);
  int idy = get_global_id(1);

  long v = 0;
  if (!(idx==0 || idx==N-1 || idy==0 || idy==M-1) ) {
    for (int i=-1; i<2; i++) {
      for (int j=-1; j<2; j++) {
        v = v + IN[(idx+i) + (idy+j)*N];
      }
    }
    v = v / 9;
  } else {
    v = IN[(idx) + (idy)*N];
  }
  OUT[idx + idy*N] = v;
}
"

*(a::(Int64, Int64), b::(Int64, Int64)) = (a[1] * b[1], a[2] * b[2])


type Benchmark <: Proc end

Base.string(::Benchmark) = "OpenCLBlur"
Base.length(p::Benchmark, n) = n
Base.isvalid(p::Benchmark, n) = n & (n-1) == 0
function Base.start(p::Benchmark, n)

  device, ctx, queue = cl.create_compute_context()
  p = cl.Program(ctx, source=kernel) |> cl.build!
  k = cl.Kernel(p, "blur")

  img = rand(Int64, n*n)

  img_in = cl.Buffer(Int64, ctx, (:r, :copy), hostbuf=img)
  img_out = cl.Buffer(Int64, ctx, :w, length(img))

  (device, ctx, queue, k, img, img_in, img_out)
end

function Base.run(p::Benchmark, n, state)

  (device, ctx, queue, k, img, img_in, img_out) = state

  gs = n
  bs = n > 32 ? 32 : n

  k[queue, (gs,gs), (bs,bs)](img_in, img_out)

end
function Base.done(p::Benchmark, n, state)

  (dev, ctx, queue, k, img, img_in, img_out) = state;

  r = cl.read(queue, img_out)

  cl.release!(ctx)

end

export Benchmark

end
