nn = Chain(Dense(21, 21, tanh), Dense(21, 21))
nn1 = FastChain(FastDense(21, 21, tanh), FastDense(21, 21))

X = randn(21, 10)
Y = rand(21, 10)
KitML.sci_train!(nn, (X, Y), ADAM())
KitML.sci_train!(nn, Flux.Data.DataLoader(X, Y), ADAM(); device = cpu, epoch = 1)
#KitML.sci_train(nn1, (X, Y))
#KitML.sci_train(nn1, (X, Y); device = cpu, save_best = true, maxiters = 200)