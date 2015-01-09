module PTX

base="/home/havard/projects/PTX.jl/julia2ptx/"
julia2ptx="julia2ptx"

include("noinline.jl")
include("OpenCL.jl")


include("gpu_array.jl")
include("module.jl")


function code_ptx(fn, args)

  code = code_module(fn, args)

  f = open(".kernel.ll", "w")
  write(f, code)
  close(f)
  readall(`$base$julia2ptx -O3 -mcpu=sm_20 .kernel.ll -o -`)

end

export
  code_ptx,

  GPUArray,
  getindex,
  setindex!

end
