using LinearAlgebra, CUDA, Flux, JLD2
import KitML

cd(@__DIR__)
data = KitML.load_data("2021-02-20_09:30:26.csv"; mode = :csv, dlm = ",")

X = hcat(data.u_0, data.u_1, data.u_2, data.u_3) |> permutedims
Y = data.h |> permutedims

data = nothing # deallocate

#nn = KitML.ICNNChain(4, 1, [16, 32, 32, 8], tanh)
@load "model.jld2" nn

loss(x, y) = sum(abs2, nn(x) - y) / 15968000

for iter = 1:1000
    KitML.sci_train!(nn, (X, Y), ADAM(); epoch = 2, batch = 4000, device = cpu)

    if iter % 2 == 0
        @show loss(X, Y)
        KitML.save_model(nn)
    end
end

#fnn = KitML.FastICNN(4, 1, [16, 64, 64, 8], tanh)
#res = KitML.sci_train(fnn, (X, Y); device = cpu, maxiters = 200)