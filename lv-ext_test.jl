using NNlib, BenchmarkTools, Metalhead, Flux
using Flux.Losses: logitcrossentropy

# create some test data
dtype = Float64 # Float64
b_size = 32
# img = rand(dtype, 50, 50, 6, b_size)
# weight = rand(dtype, 5, 5, 6, 12)
img = rand(dtype, 224, 224, 3, b_size)
weight = rand(dtype, 5, 5, 3, 24)

# dims = DenseConvDims(size(img), size(weight), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, flipkernel=true)
dims = DenseConvDims(size(img), size(weight), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, flipkernel=false) # works now

# output arrays for comparison
out_size = NNlib.output_size(dims)
# out1 = zeros(dtype, out_size..., 12, b_size)
out1 = zeros(dtype, out_size..., 24, b_size)
out2 = copy(out1)

output_gradient = rand(dtype, size(out1)...)
input_gradient1 = zeros(dtype, size(img))
input_gradient2 = zeros(dtype, size(img))
weight_gradient1 = zeros(dtype, size(weight))
weight_gradient2 = zeros(dtype, size(weight))

# without lv-ext
println("without lv-ext")

# out1 = conv!(out1, img, weight, dims)
# out1 = @time conv!(out1, img, weight, dims)
out1 = @btime conv!($out1, $img, $weight, $dims)

# input_gradient1 = NNlib.∇conv_data!(input_gradient1, output_gradient, weight, dims)
# input_gradient1 = @time NNlib.∇conv_data!(input_gradient1, output_gradient, weight, dims)
input_gradient1 = @btime NNlib.∇conv_data!($input_gradient1, $output_gradient, $weight, $dims)

# weight_gradient1 = NNlib.∇conv_filter!(weight_gradient1, img, output_gradient, dims)
# weight_gradient1 = @time NNlib.∇conv_filter!(weight_gradient1, img, output_gradient, dims)
weight_gradient1 = @btime NNlib.∇conv_filter!($weight_gradient1, $img, $output_gradient, $dims)

# with lv-ext
println("with lv-ext")

using LoopVectorization

# out2 = conv!(out2, img, weight, dims)
# out2 = @time conv!(out2, img, weight, dims)
out2 = @btime conv!($out2, $img, $weight, $dims)

# input_gradient2 = NNlib.∇conv_data!(input_gradient2, output_gradient, weight, dims)
# input_gradient2 = @time NNlib.∇conv_data!(input_gradient2, output_gradient, weight, dims)
input_gradient2 = @btime NNlib.∇conv_data!($input_gradient2, $output_gradient, $weight, $dims)

# weight_gradient2 = NNlib.∇conv_filter!(weight_gradient2, img, output_gradient, dims)
# weight_gradient2 = @time NNlib.∇conv_filter!(weight_gradient2, img, output_gradient, dims)
weight_gradient2 = @btime NNlib.∇conv_filter!($weight_gradient2, $img, $output_gradient, $dims)

# validate
@info isapprox(out1, out2)
@info isapprox(input_gradient1, input_gradient2)
@info isapprox(weight_gradient1, weight_gradient2)

#=
Some results on a Ryzen 9 5900X
without lv-ext
  61.438 ms (132 allocations: 332.36 MiB)
  84.631 ms (133 allocations: 332.36 MiB)
  146.336 ms (25 allocations: 13.85 MiB)
with lv-ext
  9.683 ms (2 allocations: 7.23 KiB)
  7.404 ms (1155 allocations: 68.16 KiB)
  120.198 ms (2 allocations: 7.23 KiB)
[ Info: true
[ Info: true 
[ Info: false # true if dtype=Float64, reason unknown
=#