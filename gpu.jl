import CUDA

base="/home/havard/projects/PTX.jl/"
julia2ptx="julia2ptx"


function create()
  dev = CUDA.CuDevice(0)
  ctx = CUDA.create_context(dev)
  (dev, ctx)
end


function void() end
@inline get_global_id(id::Int64) = get_global_id(int32(id))
@noinline get_global_id(id::Int32) = int64(id)



type GPUArray{T <: Union(Int32, Int64, Float32, Float64)}
  data::Array{T,1}
end

@noinline getindex{T}(A::GPUArray{T}, i::Int64) = A.data[i]
@noinline setindex!{T}(A::GPUArray{T}, x::T, i::Int64) = A.data[i] = x

llvm_type(::Type{Int32}) = "i32"
llvm_type(::Type{Int64}) = "i64"
llvm_type(::Type{Float32}) = "float"
llvm_type(::Type{Float64}) = "double"
llvm_type{T}(::Type{GPUArray{T}}) = llvm_type(T) * "*"

function code_lowered_array(fn, args)

  code = code_module(fn, args)

  run(`mkdir -p .spir`)
  f = open(".spir/kernel.ll", "w")
  write(f, code)
  close(f)
  readall(`$base$julia2ptx .spir/kernel.ll` .> `cat`)

end

function code_module(fn, args)

  originalSTDOUT = STDOUT
  (outRead, outWrite) = redirect_stdout()

  code_llvm(fn, args)

  close(outWrite)
  fn = readavailable(outRead)
  close(outRead)
  redirect_stdout(originalSTDOUT)

  unmangle(makeModule(fn, args))

end

function code_ptx(fn, args)
  code = code_lowered_array(fn, args)

  run(`mkdir -p .ptx`)
  f = open(".ptx/kernel.ll", "w")
  write(f, code)
  close(f)

  #run(`llvm-as .ptx/kernel.ll`)
  #run(`llvm-link .ptx/code.bc -o .ptx/kernel.bc`)

  ptx = readall(`llc -O3 -mcpu=sm_20 .ptx/kernel.ll -o -`)

  # Hack to remove .weak declaration from get_global_id
  #ptx = readall(`cat .ptx/kernel.ptx` |> `grep -v .weak`)

  return ptx
end

function makeModule(fn, args)

  jl_value = "%jl_value_t = type { %jl_value_t* }"

  declares = makeDeclares(fn)
  fn = stripDebugging(fn)
  metadata = makeMetadata(args)
 
  "$jl_value\n$fn\n$declares\n\n$metadata"
end

function makeDeclares(fn)
  decs = []
  for line in split(fn, "\n")
    m = match(r"call (.*), !dbg.*$", line)
    if m != nothing
      proto = removeValues(m.captures[1])
      push!(decs, "declare $proto")
    end
  end
  str_descs=""
  for dec in Set(decs)
     str_descs="$str_descs$dec\n"
  end
  str_descs
end

function makeMetadata(args)

  metadata = "!julia.args = !{!0}\n\n!0 = metadata !{"

  delim = ""
  for arg in args
    metadata = metadata * delim * "metadata !\"" * llvm_type(arg) * "\""
    delim = ", "
  end

  metadata * "}"

end

function removeValues(call)

    regex = r"(%?[a-zA-Z0-9]+ @\".*\"\()(.*)(\))"

    m = match(regex, call)
    if m != nothing
      start = m.captures[1]

      types = ""
      typesWithValues = m.captures[2]
      prev = ' '
      isType = true
      for ch in typesWithValues

        if ch == ' '
          if  prev == ','
            types = "$types,"
            isType = true
          else
            isType = false
          end
        end
        prev = ch

        if isType
          types = "$types$ch"
        end
      end

      "$start$types)"
    else
      ""
    end
end

function stripDebugging(fn)

  lines = ""

  for line in split(fn, "\n")
    m = match(r"(.*)(, !dbg .*)", line)
    if m == nothing
      lines = "$lines$line\n"
    else 
      stripped = m.captures[1]
      lines = "$lines$stripped\n"
    end

  end
  lines
end

function unmangle(fn)

  lines = ""

  for line in split(fn, "\n")

    m = match(r"(.*?)\"julia_(.*?)!?;.*?(\(.*)", line)
    if m == nothing
      lines = "$lines$line\n"
    else
      stripped = m.captures[1]*m.captures[2]*m.captures[3]
      lines = "$lines$stripped\n"
    end   

  end
  lines

end
