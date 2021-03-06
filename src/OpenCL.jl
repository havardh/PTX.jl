@inline get_global_id(id::Int64) = get_global_id(int32(id))
@noinline get_global_id(id::Int32) = int64(id)

@noinline barrier(::Int32) = nothing


export
  get_global_id,
  barrier
