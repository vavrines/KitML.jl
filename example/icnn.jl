using DiffEqFlux, Flux
import KitML

X = rand(4, 10)
Y = X .^ 2

# dense layers
nn = FastChain(FastDense(4, 4, tanh), FastDense(4, 4))
res = KitML.sci_train(nn, (X, Y))
res = KitML.sci_train(nn, (X, Y), res.minimizer, ADAM(); maxiters = 10000)

# fast affines
nn = FastChain(
    KitML.FastAffine(4, 4, tanh; precision = Float64),
    KitML.FastAffine(4, 4; precision = Float64),
)
res = KitML.sci_train(nn, (X, Y))
res = KitML.sci_train(nn, (X, Y), res.minimizer, ADAM(); maxiters = 10000)

# fast icnn
icnn = KitML.FastICNN(4, 4, [4, 4], tanh)
res2 = KitML.sci_train(icnn, (X, Y))
res2 = KitML.sci_train(icnn, (X, Y), res2.minimizer, ADAM(); maxiters = 10000)

# icnn
icnn2 = KitML.ICNNChain(4, 4, [4, 4], tanh)
KitML.sci_train!(icnn2, (X, Y), ADAM(), 10000, 10)
