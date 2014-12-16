using Images, Color, FixedPointNumbers



include("blur_julia.jl")
include("blur_cuda.jl")
include("blur_opencl.jl")

function run(image) 

  println(image)

end


function main()
  if length(ARGS) != 1
    println("usage: julia blur.jl <file>")
    exit(1)
  end

  file = ARGS[1]

  img = read(file)

  write(julia_run(img), "julia.png")
  write(cuda_run(img), "cuda.png")
  write(opencl_run(img), "opencl.png")

end

function read(file)
  img = convert(Array, float32(convert(Image{Gray}, imread(file))))
  vec(map(function(c) convert(Float32, c) end, img))
end

function write(image, file)

  imwrite(convert(Image{Gray}, reshape(image, 256, 256)), file)

end




main()
