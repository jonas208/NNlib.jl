module NNlibLoopVectorizationExt

using NNlib
using LoopVectorization
using Random, Statistics

include("conv.jl")

println("lv-ext loaded")

#=
using NNlib, BenchmarkTools, Metalhead

# create some test data
b_size = 32
img = rand(Float32, 224, 224, 3, b_size)
weight = rand(Float32, 5, 5, 3, 9)

dims = DenseConvDims(size(img), size(weight), stride=(2, 1), padding=(0, 0), dilation=(2, 1), groups=1, flipkernel=true)
# flipkernel=false is currently wrong! GradValley's conv can only perform Flux' CrossCorr (flipkernel=true)
# dims = DenseConvDims(size(img), size(weight), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, flipkernel=false)

# output arrays for comparison
out_size = NNlib.output_size(dims)
out1 = zeros(Float32, out_size..., 9, b_size)
out2 = copy(out1)

output_gradient = rand(Float32, size(out1)...)
input_gradient1 = zeros(Float32, size(img))
input_gradient2 = zeros(Float32, size(img))
weight_gradient1 = zeros(Float32, size(weight))
weight_gradient2 = zeros(Float32, size(weight))

# without lv-ext

out1 = @btime conv!($out1, $img, $weight, $dims)
input_gradient1 = @btime NNlib.∇conv_data!(input_gradient1, output_gradient, weight, dims)

model = ResNet(34; pretrain=false)
@btime model(img)

# with lv-ext

using LoopVectorization

out2 = @btime conv!($out2, $img, $weight, $dims)

input_gradient2 = @btime NNlib.∇conv_data!(input_gradient2, output_gradient, weight, dims)

@btime model(img)

# validate
@info isapprox(out1, out2)
@info isapprox(input_gradient1, input_gradient2)

#=
Some results on an Ryzen 9 5900X
  23.392 ms (132 allocations: 163.17 MiB)
  39.256 ms (133 allocations: 163.17 MiB)
  1.572 s (5885 allocations: 4.18 GiB)
lv-ext loaded
  5.285 ms (0 allocations: 0 bytes)
  5.927 ms (0 allocations: 0 bytes)
  1.294 s (1410 allocations: 2.28 GiB)
[ Info: true
[ Info: true
=#
=#

end # module