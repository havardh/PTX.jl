using OpenCL
const cl = OpenCL

const kernel = "
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define N 256
#define M 256

__kernel void blur(__global const float *IN,
                   __global float *OUT)
{
  int idx = get_global_id(0);
  int idy = get_global_id(1);

  float v = 0;
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
}"


function opencl_run(image)
  n = 256
  device, ctx, queue = cl.create_compute_context()
  p = cl.Program(ctx, source=kernel) |> cl.build!
  k = cl.Kernel(p, "blur")

  img_in = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf=image)
  img_out = cl.Buffer(Float32, ctx, :w, length(image))

  gs = 256
  bs = 16

  k[queue, (gs,gs), (bs,bs)](img_in, img_out)

  result = cl.read(queue, img_out)
  cl.release!(ctx)

  return result
end

