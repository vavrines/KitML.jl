using Test, Flux, DiffEqFlux
import KitML

begin
    u = collect(-5:0.5:5)
    ω = ones(21) ./ 21
    prim = [1.0, 0.0, 1.0]
    w = KitML.KitBase.prim_conserve(prim, 5/3)
    M = KitML.KitBase.maxwellian(u, prim)
    τ = 0.01
    f = deepcopy(M)
    nn = Chain(Dense(21, 21, tanh), Dense(21, 21))
    p = initial_params(nn)
end

# IO
cd(@__DIR__)
KitML.load_data("dataset.csv", dlm = ",")
KitML.save_model(nn)

# Layer
sm = KitML.Shortcut(nn)
sm(rand(21))
show(sm)

# Train
X = randn(21, 10)
Y = rand(21, 10)
KitML.sci_train!(nn, (X, Y), ADAM(), 1, 1)
KitML.sci_train!(nn, Flux.Data.DataLoader(X, Y), ADAM(), 1)

fnn = FastChain(FastDense(21, 21, tanh), FastDense(21, 21))
KitML.sci_train(fnn, (X, Y))

# Equation
df = KitML.ube_dfdt(f, (M, τ, (nn, p)), 0.1)
df2 = similar(df)
KitML.ube_dfdt!(df2, f, (M, τ, (nn, p)), 0.1)
@test df ≈ df2 atol = 0.01

nn1 = FastChain(FastDense(21, 21, tanh), FastDense(21, 21, tanh))
p1 = initial_params(nn1)
KitML.ube_dfdt(f, (M, τ, (nn1, p1)), 0.1)
KitML.ube_dfdt!(similar(f), f, (M, τ, (nn1, p1)), 0.1)

# Solver
fw = zeros(3)
fh = zeros(21)

nn2 = Chain(Dense(42, 42, tanh), Dense(42, 42, tanh))
p2 = initial_params(nn2)

KitML.step_ube!(fw, fh, fh, w, prim, f, f, fw, fh, fh, u, ω, (2.0, 5/3, 0.01, 0.5, 1.0, 0.1, 0.001, zeros(3), zeros(3), nn2, p2); mode=:bgk)
KitML.step_ube!(fw, fh, fh, w, prim, f, f, fw, fh, fh, u, ω, (2.0, 5/3, 0.01, 0.5, 1.0, 0.1, 0.001, zeros(3), zeros(3), nn2, p2); mode=:shakhov)
KitML.step_ube!(fw, fh, fh, w, prim, f, f, fw, fh, fh, u, ω, (2.0, 5/3, 0.01, 0.5, 1.0, 0.1, 0.001, zeros(3), zeros(3), nn2, p2); mode=:nn)

# Closure
m = KitML.create_neural_closure(10, 5)
X = randn(10, 2)
Y = randn(5, 2)
KitML.train_neural_closure(X, Y, m, 2)
KitML.neural_closure(rand(10), m)

nn2 = KitML.ICNNLayer(5, 5, 1, tanh)
nn2(randn(5))

icnn = KitML.ICNN(10, [10, 5, 1], tanh)

icnn(randn(10))