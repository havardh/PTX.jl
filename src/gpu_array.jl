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
