
import CUDA
using PTX

import OpenCL
const cl = OpenCL


const kernel = "
__kernel void blur(__global const long *IN,
                   __global long *OUT,
                   const long n)
{

  int idx = get_global_id(0);
  int idy = get_global_id(1);

  long v = 0;
  if (!(idx==0 || idx==n-1 || idy==0 || idy==n-1) ) {
    for (int i=-1; i<2; i++) {
      for (int j=-1; j<2; j++) {
        v = v + IN[(idx+i) + (idy+j)*n];
      }
    }
    v = v / 9;
  } else {
    v = IN[(idx) + (idy)*n];
  }
  OUT[idx + idy*n] = v;
}
"

T = Int64

@inline IDX(i,j,n) = i+j*n
@inline ON_BORDER(i,j,n) = (i==0 || j==0 || i==n-1 || j==n-1)

type Result
  n::Int
  setup::Float64
  exec::Float64
  teardown::Float64
end

# Kernel
function blur(IN::GPUArray{T}, OUT::GPUArray{T}, n::Int64)

  idx = get_global_id(0)
  idy = get_global_id(1)

  v = 0

  if !ON_BORDER(idx, idy,n)
    for i=-1:1, j=-1:1
      v = v + IN[IDX(idx+i, idy+j,n)]
    end
    v = div(v, 9)
  else
    v = IN[IDX(idx, idy,n)]
  end

  OUT[IDX(idx, idy,n)] = v

  return
end

const ptx = code_ptx(blur, (GPUArray{T}, GPUArray{T}, Int64))

dev = CUDA.CuDevice(0)
cu_ctx = CUDA.create_context(dev)

md1 = CUDA.CuModule(source=ptx)
julia_cuda_kernel = CUDA.CuFunction(md1, "blur")

md2 = CUDA.CuModule(filename="kernel/blur.ptx")
cuda_kernel = CUDA.CuFunction(md2, "blur")


#######p2 = cl.Program(cl_ctx, binaries=[ptx]) |> cl.build!
#julia_opencl_kernel = cl.Kernel(p2, "blur")


function cuda_run(kernel, input, n)
  gc()
  copy_to = @elapsed begin
    img_in = CUDA.CuArray(input)
    img_out = CUDA.CuArray(T, n*n)
    CUDA.ctx_synchronize()
  end

  gs = n > 32 ? div(n, 32) : 1
  bs = n > 32 ? 32 : n
  gc()
  execution = @elapsed begin
    CUDA.launch(kernel, (gs, gs), (bs, bs), (img_in, img_out, n))
    CUDA.ctx_synchronize()
  end
  gc()
  copy_from = @elapsed begin
    result = CUDA.to_host(img_out)
    CUDA.free(img_in)
    CUDA.free(img_out)
    CUDA.ctx_synchronize()
  end

  Result(n, copy_to, execution, copy_from)
end

function opencl_run(ctx, queue, kernel, input, n)

  gc()
  copy_to = @elapsed begin
    img_in = cl.Buffer(T, ctx, (:r, :copy), hostbuf=input)
    img_out = cl.Buffer(T, ctx, :w, length(input))
    cl.finish(queue)
  end

  gs = n
  bs = n > 32 ? 32 : n
  gc()
  execution = @elapsed begin
    kernel[queue, (gs,gs), (bs,bs)](img_in, img_out, n)
    cl.finish(queue)
  end
  gc()
  copy_from = @elapsed begin
    r = cl.read(queue, img_out)

    #cl.release!(img_in)
    #cl.release!(img_out)
    cl.finish(queue)
  end


  Result(n, copy_to, execution, copy_from)
end


function average_result(n, all_result)
  result = Result(n, 0.0, 0.0, 0.0)
  for res in all_result
    result.setup += res.setup
    result.exec += res.exec
    result.teardown += res.teardown
  end
  result.setup /= m
  result.exec /= m
  result.teardown /= m
  return result
end

julia_cuda_results = Any[]
#julia_opencl_results = Any[]
cuda_results = Any[]
opencl_results = Any[]

const m = 1
device, cl_ctx, queue = cl.create_compute_context()
p1 = cl.Program(cl_ctx, source=kernel) |> cl.build!
opencl_kernel = cl.Kernel(p1, "blur")
opencl_run(cl_ctx, queue, opencl_kernel, rand(T, 4*4), 4)
cl.release!(cl_ctx)

cuda_run(julia_cuda_kernel, rand(T, 4*4), 4)
cuda_run(cuda_kernel, rand(T, 4*4), 4)

for i=2^8:128:9344
  n = i

  device, cl_ctx, queue = cl.create_compute_context()
  p1 = cl.Program(cl_ctx, source=kernel) |> cl.build!
  opencl_kernel = cl.Kernel(p1, "blur")


  julia_cuda_all_result = Result[]
  julia_opencl_all_result = Result[]
  cuda_all_result = Result[]
  opencl_all_result = Result[]
  tot = @elapsed for j=1:m

    input = rand(T, n*n)
   
    push!(julia_cuda_all_result, cuda_run(julia_cuda_kernel, input, n))
    #push!(julia_opencl_all_result, julia_opencl_run(input, n))    
    push!(cuda_all_result, cuda_run(cuda_kernel, input, n))
    push!(opencl_all_result, opencl_run(cl_ctx, queue, opencl_kernel, input, n))

  end
  println(n, ": ", tot)
  push!(julia_cuda_results, average_result(n, julia_cuda_all_result))
  #push!(julia_opencl_results, average_result(n, julia_opencl_all_result))
  push!(cuda_results, average_result(n, cuda_all_result))
  push!(opencl_results, average_result(n, opencl_all_result))

  cl.release!(cl_ctx)


end

CUDA.destroy(cu_ctx)

write_total(f, r) = @printf(f, "%d %1.6f\n", r.n, r.setup + r.exec + r.teardown)
write_exec(f, r) = @printf(f, "%d %1.6f\n", r.n, r.exec)
write_memory(f, r) = @printf(f, "%d %1.6f\n", r.n, r.setup + r.teardown)

function write(name, results, write_f)
  f = open(name, "w")
  for r in results
    write_f(f, r)
  end
  close(f)
end

#write("result/cpu-total.dat", cpu_results, write_total)
write("result/julia-cuda-total.dat", julia_cuda_results, write_total)
write("result/cuda-total.dat", cuda_results, write_total)
write("result/opencl-total.dat", opencl_results, write_total)

#write("result/cpu-exec.dat", cpu_results, write_exec)
write("result/julia-cuda-exec.dat", julia_cuda_results, write_exec)
write("result/cuda-exec.dat", cuda_results, write_exec)
write("result/opencl-exec.dat", opencl_results, write_exec)

#write("result/cpu-mem.dat", cpu_results, write_memory)
write("result/julia-cuda-mem.dat", julia_cuda_results, write_memory)
write("result/cuda-mem.dat", cuda_results, write_memory)
write("result/opencl-mem.dat", opencl_results, write_memory)


