#--- 1D ---#
begin
    set = Setup(space = "1d1f1v")
    ps = PSpace1D(0.0, 1.0, 10)
    vs = VSpace1D(-5.0, 5.0, 24)
    gas = Gas(K = 0.0)
    
    prim0 = [1.0, 0.0, 1.0]
    fw = (args...) -> prim_conserve(prim0, gas.γ)
    ff = function(args...)
        prim = conserve_prim(fw(args...), gas.γ)
        h = maxwellian(vs.u, vs.v, prim)
        return h
    end
    bc = function(x, y)
        return prim0
    end
    ib = IB1F(fw, ff, bc)

    ks = SolverSet(set, ps, vs, gas, ib)
end

w = [1.0, 0.0, 1.0]
prim = conserve_prim(w, 3.0)
f = maxwellian(ks.vs.u, prim)
regime_data(ks, w, zeros(3), f)

#--- 2D ---#
begin
    set = Setup(
        case = "cylinder",
        space = "2d2f2v",
        boundary = ["maxwell", "extra", "mirror", "mirror"],
        limiter = "minmod",
        cfl = 0.5,
        maxTime = 15.0, # time
    )
    ps = CSpace2D(1.0, 6.0, 60, 0.0, π, 50, 1, 1)
    vs = VSpace2D(-10.0, 10.0, 48, -10.0, 10.0, 48)
    gas = Gas(Kn = 1e-2, Ma = 5.0, K = 1.0)
    
    prim0 = [1.0, 0.0, 0.0, 1.0]
    prim1 = [1.0, gas.Ma * sound_speed(1.0, gas.γ), 0.0, 1.0]
    fw = (args...) -> prim_conserve(prim1, gas.γ)
    ff = function(args...)
        prim = conserve_prim(fw(args...), gas.γ)
        h = maxwellian(vs.u, vs.v, prim)
        b = h .* gas.K / 2 / prim[end]
        return h, b
    end
    bc = function(x, y)
        if abs(x^2 + y^2 - 1) < 1e-3
            return prim0
        else
            return prim1
        end
    end
    ib = IB2F(fw, ff, bc)

    ks = SolverSet(set, ps, vs, gas, ib)
end

w = [1.0, 0.0, 0.0, 1.0]
prim = conserve_prim(w, 5/3)
f = maxwellian(ks.vs.u, ks.vs.v, prim)
regime_data(ks, w, zeros(4), zeros(4), f)
