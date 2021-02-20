using LinearAlgebra, CUDA, Flux
import KitML

cd(@__DIR__)
data = KitML.load_data("2021-02-20_09:30:26.csv"; mode = :csv, dlm = ",")

X = hcat(data.u_0, data.u_1, data.u_2, data.u_3) |> permutedims
Y = data.h |> permutedims

data = nothing # deallocate

nn = KitML.ICNNChain(4, 1, [16, 64, 64, 8], tanh)
KitML.sci_train!(nn, (X, Y), ADAM(), 2, 256)

fnn = KitML.FastICNN(4, 1, [16, 64, 64, 8], tanh)
res = KitML.sci_train(fnn, (X, Y); device = cpu)