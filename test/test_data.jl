@info "testing data handlers"

using KitML.KitBase

#--- 1D ---#
begin
    set = Setup(space = "1d1f1v")
    ps = PSpace1D(0.0, 1.0, 10)
    vs = VSpace1D(-5.0, 5.0, 16)
    gas = Gas(K = 0.0)
    ib = nothing

    ks = SolverSet(set, ps, vs, gas, ib)
end

w = [1.0, 0.0, 1.0]
prim = conserve_prim(w, 3.0)
f = maxwellian(ks.vs.u, prim)
regime_data(ks, w, zeros(3), f)


#--- 2D ---#
begin
    set = Setup(space = "2d2f2v")
    ps = PSpace1D(0.0, 1.0, 10)
    vs = VSpace2D(-10.0, 10.0, 24, -10.0, 10.0, 24)
    gas = Gas(K = 1.0)
    ib = nothing

    ks = SolverSet(set, ps, vs, gas, ib)
end

w = [1.0, 0.0, 0.0, 1.0]
prim = conserve_prim(w, 5 / 3)
f = maxwellian(ks.vs.u, ks.vs.v, prim)
regime_data(ks, w, zeros(4), zeros(4), f)

#--- 3D ---#
begin
    set = Setup(space = "1d1f3v")
    ps = PSpace1D(0.0, 1.0, 10)
    vs = VSpace3D(-10, 10, 16, -10, 10, 16, -10, 10, 16)
    gas = Gas(K = 0.0)
    ib = nothing

    ks = SolverSet(set, ps, vs, gas, ib)
end

w = [1.0, 0.0, 0.0, 0.0, 1.0]
prim = conserve_prim(w, 5 / 3)
f = maxwellian(ks.vs.u, ks.vs.v, ks.vs.w, prim)
regime_data(ks, w, zeros(5), zeros(5), zeros(5), f)
