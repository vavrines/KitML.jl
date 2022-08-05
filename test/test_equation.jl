@info "testing equations"

begin
    u = collect(-5:0.5:5)
    ω = ones(21) ./ 21
    prim = [1.0, 0.0, 1.0]
    w = KitML.KitBase.prim_conserve(prim, 5 / 3)
    M = KitML.KitBase.maxwellian(u, prim)
    τ = 0.01
    f = deepcopy(M)
    nn = Chain(Dense(21, 21, tanh), Dense(21, 21))
    p = init_params(nn)
    nn1 = FnChain(FnDense(21, 21, tanh), FnDense(21, 21))
    p1 = init_params(nn1)
end

df = KitML.ube_dfdt(f, (M, τ, (nn, p)), 0.1)
df2 = similar(df)
KitML.ube_dfdt!(df2, f, (M, τ, (nn, p)), 0.1)
@test df ≈ df2 atol = 0.01

KitML.ube_dfdt(f, (M, τ, (nn1, p1)), 0.1)
KitML.ube_dfdt!(similar(f), f, (M, τ, (nn1, p1)), 0.1)