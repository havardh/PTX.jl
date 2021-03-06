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

function makeModule(fn, args)

  target = """
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "nvptx64-nvidia-cuda"
"""
  jl_value = "%jl_value_t = type { %jl_value_t* }"

  declares = makeDeclares(fn)
  fn = stripDebugging(fn)
  metadata = makeMetadata(args)
 
  "$target\n$jl_value\n$fn\n$declares\n\n$metadata"
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
