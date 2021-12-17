@info "testing closures"

KitML.create_neural_closure(10, 5)

nn = Chain(Dense(4, 4, tanh), Dense(4, 1))
KitML.neural_closure(nn, randn(Float32, 4))

fnn = FastChain(FastDense(4, 4, tanh), FastDense(4, 1))
p = initial_params(fnn)
KitML.neural_closure(fnn, p, randn(Float32, 4))

cd(@__DIR__)
#model = KitML.load_model("tfmodel.h5"; mode = :tf)
#KitML.neural_closure(model, randn(Float32, 1, 4))
