module CPUMM

using BenchmarkLite

type Benchmark{T} <: Proc end

Base.string{T}(::Benchmark{T}) = "Julia CPU(" * string(T) * ")"
Base.length{T}(p::Benchmark{T}, n) = n
Base.isvalid{T}(p::Benchmark{T}, n) = n > 0

function Base.start{T}(p::Benchmark{T}, n)

  A = rand(T, n, n);
  B = rand(T, n, n);
  
  (A, B)

end

function Base.run{T}(p::Benchmark{T}, n, state)

  (A, B) = state

  C = A * B

end

Base.done{T}(p::Benchmark{T}, n, state) = nothing

export Benchmark

end
