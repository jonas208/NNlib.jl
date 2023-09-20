using NNlib, BenchmarkTools, Metalhead, Flux
using Flux.Losses: logitcrossentropy

model = ResNet(34; pretrain=false)

for i in 1:2
  @time begin
    gs, _ = gradient(model, img) do m, x  # calculate the gradients
    logitcrossentropy(m(x), Flux.onehotbatch(rand(1:1000, b_size), 1:1000)) 
    end
  end
end
model(img)
out_model1 = @time model(img)

for i in 1:2
  @time begin
    gs, _ = gradient(model, img) do m, x  # calculate the gradients
    logitcrossentropy(m(x), Flux.onehotbatch(rand(1:1000, b_size), 1:1000)) 
    end
  end
end
model(img)
out_model2 = @time model(img)

@info isapprox(out_model1, out_model2)