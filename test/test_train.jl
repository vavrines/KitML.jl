nn = Chain(Dense(21, 21, tanh), Dense(21, 21))
nn1 = FastChain(FastDense(21, 21, tanh), FastDense(21, 21))

X = randn(Float32, 21, 10)
Y = rand(Float32, 21, 10)
KitML.sci_train!(nn, (X, Y), ADAM())
KitML.sci_train!(nn, Flux.Data.DataLoader((X, Y)), ADAM(); device = cpu, epoch = 1)
KitML.sci_train(nn1, (X, Y))

loss(p) = nn1(X, p) |> sum
p1 = initial_params(nn1)
KitML.sci_train(loss, p1)

cd(@__DIR__)
model = KitML.load_model("tfmodel.h5"; mode = :tf)
KitML.sci_train!(model, (randn(Float32, 1, 4), randn(Float32, 1, 1)))