using BenchmarkLite

@inline IDX(i,j) = i+j*m
@inline ON_BORDER(i,j) = (i==0 || j==0 || i==n-1 || j==m-1)
@inline ON_BORDER_HOST(i, j) = ON_BORDER(i-1,j-1)

# Host implementation
function blur(img::Array{Int64})

  out = zeros(Int64, n*m)

  for idx = 1:length(img)/n, idy = 1:length(img)/m

    if !ON_BORDER_HOST(idx, idy)
      v = 0
      for i=-1:1, j=-1:1
        v = v + img[((idx+i) + (idy+j-1)*n)]
      end
      out[(idx + (idy-1)*n)] = div(v, 9)
    else
      out[(idx + (idy-1)*n)] = img[((idx) + (idy-1)*n)]
    end

  end

  return out

end
type Benchmark <: Proc end

Base.string(::Benchmark) = "CPUBlur"
Base.length(p::Benchmark, n) = n
Base.isvalid(p::Benchmark, n) = n & (n-1) == 0
function Base.start(p::Benchmark, n)

  img = rand(Int64, n*n)

  img
end

function Base.run(p::Benchmark, n, state)

  img = state


  blur(img)

end
Base.done(p::Benchmark, n, state) = nothing


export Benchmark

end

