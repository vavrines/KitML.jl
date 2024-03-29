@info "testing solvers"

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
    fw = zeros(3)
    fh = zeros(21)
    nn2 = Chain(Dense(42, 42, tanh), Dense(42, 42, tanh))
    p2 = init_params(nn2)
end

KitML.step_ube!(
    fw,
    fh,
    fh,
    w,
    prim,
    f,
    f,
    fw,
    fh,
    fh,
    u,
    ω,
    (2.0, 5 / 3, 0.01, 0.5, 1.0, 0.1, 0.001, zeros(3), zeros(3), nn2, p2);
    mode = :bgk,
)
KitML.step_ube!(
    fw,
    fh,
    fh,
    w,
    prim,
    f,
    f,
    fw,
    fh,
    fh,
    u,
    ω,
    (2.0, 5 / 3, 0.01, 0.5, 1.0, 0.1, 0.001, zeros(3), zeros(3), nn2, p2);
    mode = :shakhov,
)
KitML.step_ube!(
    fw,
    fh,
    fh,
    w,
    prim,
    f,
    f,
    fw,
    fh,
    fh,
    u,
    ω,
    (2.0, 5 / 3, 0.01, 0.5, 1.0, 0.1, 0.001, zeros(3), zeros(3), nn2, p2);
    mode = :nn,
)
