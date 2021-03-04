Chain(4, 4, tanh)

KitML.dense_layer(4, 4; isBias = true)
KitML.dense_layer(4, 4; isBias = false)

faf = KitML.FastAffine(4, 4, tanh)
DiffEqFlux.paramlength(faf)
DiffEqFlux.initial_params(faf)

nn = Chain(Dense(21, 21, tanh), Dense(21, 21))
sm = KitML.Shortcut(nn)
show(sm)
sm(rand(21))

icnnl = KitML.ICNNLayer(4, 4, 1, identity; fw = randn, fb = zeros, precision = Float32)
icnnc =
    KitML.ICNNChain(4, 1, [10, 10], identity; fw = randn, fb = zeros, precision = Float32)
show(icnnl)
show(icnnc)
icnnl(randn(4))
icnnl(randn(4), randn(4))
icnnc(randn(4))

fil = KitML.FastIC(4, 4, 4)
fic = KitML.FastICNN(4, 1, [10, 10])
fil(rand(4), rand(4), initial_params(fil))
fic(rand(4), initial_params(fic))