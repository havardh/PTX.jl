@inline function noinline(returnvalue)

  fn = "noinline.jl:noinline"
  msg = "This function should not be called"
  println(fn * msg)
  exit(1)

  returnvalue
end

macro noinline(ex)

  signature = ex.args[1]
  body = ex.args[2]

  @eval $signature = noinline($body)
end
