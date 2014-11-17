module CPUMM

using BenchmarkLite

type Benchmark <: Proc end

Base.string(::Benchmark) = "Julia CPU"
Base.length(p::Benchmark, n) = n
Base.isvalid(p::Benchmark, n) = n > 0

function Base.start(p::Benchmark, n)

  A = rand(Int64, n, n);
  B = rand(Int64, n, n);
  
  (A, B)

end

function Base.run(p::Benchmark, n, state)

  (A, B) = state

  C = A * B

end

Base.done(p::Benchmark, n, state) = nothing

export Benchmark

end
