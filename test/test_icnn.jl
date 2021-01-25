using Flux
import KitML

a = KitML.ICNNLayer(4, 4, 1, identity; fw = randn, fb = zeros, precision = Float32)
c = KitML.ICNNChain(4, 1, [10, 10], identity; fw = randn, fb = zeros, precision = Float32)

ne = 4
nn = KitML.create_neural_closure(ne, ne)

X = randn(4, 100)
Y = randn(4, 100)

KitML.sci_train!(nn, (X, Y), ADAM(), 1, 1)
