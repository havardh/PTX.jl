
T = Float32
const m = 2
problem_size = 3500:16:3750

const CLK_LOCAL_MEM_FENCE = int32(1)
const CLK_GLOBAL_MEM_FENCE = int32(2)

import CUDA
using PTX

@inline function computeCell{T}(A::GPUArray{T}, B::GPUArray{T}, C::GPUArray{T}, row, col, n)

  v = zero(T);
  if row < n && col < n
    i = 0;
    while i < n
      v = v + (A[i + row*n] * B[col + i*n]);
      i += 1;
    end
  end
  return v
end

function MatrixMultiply{T}(A::GPUArray{T}, B::GPUArray{T}, C::GPUArray{T}, n)

  col = get_global_id(int32(0));
  row = get_global_id(int32(1));

  C[col + (row)*n] = computeCell(A, B, C, row, col, n);

  return
end

type Result
  n::Int
  setup::Float64
  exec::Float64
  teardown::Float64
  res::Array{T,2}
end

function cpu_run(Ain, Bin, n)

  A = reshape(Ain, n, n)'
  B = reshape(Bin, n, n)'

  gc()
  execution = @elapsed C = A*B
  Result(n, 0, execution, 0, C)

end


function cuda_run(kernel, A, B, n)
  gc()
  copy_to = @elapsed begin
    A_gpu = CUDA.CuArray(A)
    B_gpu = CUDA.CuArray(B)
    C_gpu = CUDA.CuArray(T, n*n)
    CUDA.ctx_synchronize()
  end

  bs = 16
  gs = n > bs ? int(n / bs) : 1


  #println(n, ": ", gs, " ", bs, " ")

  gc()
  println("launching", kernel)
  execution = @elapsed begin
    CUDA.launch(kernel, (gs, gs), (bs, bs), (A_gpu, B_gpu, C_gpu, n))
    CUDA.ctx_synchronize()
  end
  gc()
  copy_from = @elapsed begin
   result = CUDA.to_host(C_gpu)
   CUDA.free(A_gpu)
   CUDA.free(B_gpu)
   CUDA.free(C_gpu)
   CUDA.ctx_synchronize()
  end

  Result(n, copy_to, execution, copy_from, reshape(result, n, n)')
end

dev = CUDA.CuDevice(0)
cu_ctx = CUDA.create_context(dev)
ptx = code_ptx(MatrixMultiply, (GPUArray{T}, GPUArray{T}, GPUArray{T}, Int64))
md1 = CUDA.CuModule(source=ptx)
julia_cuda_kernel = CUDA.CuFunction(md1, "MatrixMultiply")
md2 = CUDA.CuModule(filename="kernel/"*string(T)*".ptx")
cuda_kernel = CUDA.CuFunction(md2, "MatrixMultiply")

const test_size = 32

md2 = CUDA.CuModule(filename="kernel/"*string(T)*".ptx")
cuda_kernel = CUDA.CuFunction(md2, "MatrixMultiply")
#cpu_run(rand(T, test_size*test_size), rand(T, test_size*test_size), test_size)
#julia_opencl_run(rand(Int64, test_size*test_size), test_size)
#cuda_run(cuda_kernel, rand(T, test_size*test_size), rand(T, test_size*test_size), test_size)
cuda_run(julia_cuda_kernel, rand(T, test_size*test_size), rand(T, test_size*test_size), test_size)
#opencl_run(rand(Int64, test_size*test_size), test_size)




function average_result(n, all_result)

  m1 = length(all_result)

  r1 = Result(n, 0.0, 0.0, 0.0, zeros(T,0,0))
  for res in all_result
    r1.setup += res.setup
    r1.exec += res.exec
    r1.teardown += res.teardown
  end
  r1.setup /= m1
  r1.exec /= m1
  r1.teardown /= m1

  return r1

  m2 = 0
  r2 = Result(n, 0.0, 0.0, 0.0, zeros(T,0,0))
  for res in all_result

    if (abs(res.setup - r1.setup) < 0.0001 || abs(res.teardown - r1.teardown) < 0.0001)
      r2.setup += res.setup
      r2.exec += res.exec
      r2.teardown += res.teardown
      m2 += 1
    end
  end
  r2.setup /= m2
  r2.exec /= m2
  r2.teardown /= m2

  return r2
end

cpu_results = Any[]
julia_cuda_results = Any[]
#julia_opencl_results = Any[]
cuda_results = Any[]
#opencl_results = Any[]

for n=problem_size

  cpu_all_result = Result[]
  julia_cuda_all_result = Result[]
  #julia_opencl_all_result = Result[]
  cuda_all_result = Result[]
  #opencl_all_result = Result[]
  tot = @elapsed for j=1:m
    A = rand(T, n*n)
    B = rand(T, n*n)

    gc()

    #cpu_result = cpu_run(A, B, n)
    #cuda_result = cuda_run(cuda_kernel, A, B, n)
    julia_cuda_result = cuda_run(julia_cuda_kernel, A, B, n)

    j_err = 0
    c_err = 0
    for i=1:n, j=1:n
      if (abs(cpu_result.res[i,j] - julia_cuda_result.res[i,j]) > 0.0001)
        j_err += 1
      end
      if (abs(cpu_result.res[i,j] - cuda_result.res[i,j]) > 0.0001)
        c_err += 1
      end

    end
    #println(n, ": ", j_err, ", ", c_err)

    push!(cpu_all_result, cpu_result)
    push!(julia_cuda_all_result, julia_cuda_result)
    push!(cuda_all_result, cuda_result)
  end
  println(n, ": ", tot)
  push!(cpu_results, average_result(n, cpu_all_result))
  push!(julia_cuda_results, average_result(n, julia_cuda_all_result))
  #push!(julia_opencl_results, average_result(n, julia_opencl_all_result))
  push!(cuda_results, average_result(n, cuda_all_result))
  #push!(opencl_results, average_result(n, opencl_all_result))

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


write("result/"*string(T)*"-cpu-total.dat", cpu_results, write_total)
write("result/"*string(T)*"-julia-cuda-total.dat", julia_cuda_results, write_total)
write("result/"*string(T)*"-cuda-total.dat", cuda_results, write_total)
#write("result/"*string(T)*"-opencl", opencl_results, write_total)

write("result/"*string(T)*"-cpu-exec.dat", cpu_results, write_exec)
write("result/"*string(T)*"-julia-cuda-exec.dat", julia_cuda_results, write_exec)
write("result/"*string(T)*"-cuda-exec.dat", cuda_results, write_exec)
#write("result/"*string(T)*"-opencl", opencl_results, write_exec)

write("result/"*string(T)*"-cpu-mem.dat", cpu_results, write_memory)
write("result/"*string(T)*"-julia-cuda-mem.dat", julia_cuda_results, write_memory)
write("result/"*string(T)*"-cuda-mem.dat", cuda_results, write_memory)
#write("opencl", opencl_results, write_memory)

#cl.release!(cl_ctx)
#CUDA.destroy(cu_ctx)

