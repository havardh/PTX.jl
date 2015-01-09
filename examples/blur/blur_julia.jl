import CUDA
using PTX

const n = 256

@inline IDX(i,j) = (i+(j*n))
@inline ON_BORDER(i,j) = (i==0 || j==0 || i==(n-1) || j==(n-1))
@inline ON_BORDER_HOST(i, j) = ON_BORDER(i-1,j-1)


function blur(IN, OUT)

  idx = get_global_id(0)
  idy = get_global_id(1)

  v = float32(0)

  if !ON_BORDER(idx, idy)
    for i=-1:1, j=-1:1
      v = v + IN[IDX(idx+i, idy+j)]
    end
    v = v / float32(9)
  else
    v = IN[IDX(idx, idy)]
  end

  OUT[IDX(idx, idy)] = v

  return
end

function julia_run(image)

  dev = CUDA.CuDevice(0)
  ctx = CUDA.create_context(dev)

  ptx = code_ptx(blur, (GPUArray{Float32}, GPUArray{Float32}))
  
  md = CUDA.CuModule(source=ptx)
  kernel = CUDA.CuFunction(md, "blur")

  img = GPUArray{Float32}(image)

  img_in = CUDA.CuArray(img.data)
  img_out = CUDA.CuArray(Float32, n * n)

  gs = 16
  bs = 16

  CUDA.launch(kernel, (gs, gs), (bs, bs), (img_in, img_out))

  result = CUDA.to_host(img_out)

  CUDA.free(img_in)
  CUDA.free(img_out)

  CUDA.destroy(ctx)

  return result
end
