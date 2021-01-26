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
    nn1 = FastChain(FastDense(21, 21, tanh), FastDense(21, 21))
    p1 = initial_params(nn1)
end

# IO
cd(@__DIR__)
KitML.load_data("dataset.csv", dlm = ",")
KitML.save_model(nn)

# Layer
Chain(4, 4, tanh)
KitML.dense_layer(4, 4; isBias = true)
KitML.dense_layer(4, 4; isBias = false)

sm = KitML.Shortcut(nn)
sm(rand(21))
show(sm)

icnnl = KitML.ICNNLayer(4, 4, 1, identity; fw = randn, fb = zeros, precision = Float32)
icnnc = KitML.ICNNChain(4, 1, [10, 10], identity; fw = randn, fb = zeros, precision = Float32)
show(icnnl)
show(icnnc)

# Train
X = randn(21, 10)
Y = rand(21, 10)
KitML.sci_train!(nn, (X, Y), ADAM(), 1, 1)
KitML.sci_train!(nn, Flux.Data.DataLoader(X, Y), ADAM(), 1)
KitML.sci_train(nn1, (X, Y))

# Equation
df = KitML.ube_dfdt(f, (M, τ, (nn, p)), 0.1)
df2 = similar(df)
KitML.ube_dfdt!(df2, f, (M, τ, (nn, p)), 0.1)
@test df ≈ df2 atol = 0.01

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
m = KitML.create_neural_closure(10, 5; mode = :icnn)
m = KitML.create_neural_closure(10, 5; mode = :dense)
X = randn(10, 2)
Y = randn(5, 2)
KitML.neural_closure(rand(10), m)
