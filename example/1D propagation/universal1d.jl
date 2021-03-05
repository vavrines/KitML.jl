using Flux, KitBase, KitBase.FastGaussQuadrature, ProgressMeter, Plots, LinearAlgebra, Optim
import KitML

cd(@__DIR__)
include("../math.jl")

function ComputeSphericalBasisAnalytical(quadpts::AbstractVector{<:AbstractFloat})
    # Hardcoded solution for L = 1, spatialDim = 3
    monomialBasis = zeros(2,length(quadpts))
    for idx_quad in 0:(size(quadpts)[1]-1)
        monomialBasis[1,idx_quad+1]  = 1
        monomialBasis[2,idx_quad+1]  = quadpts[idx_quad+1,1] # x
    end

    return monomialBasis
end

function flux_wall!(
    ff::T1,
    f::T2,
    u::T3,
    dt,
    rot = 1,
) where {
    T1<:AbstractVector{<:AbstractFloat},
    T2<:AbstractVector{<:AbstractFloat},
    T3<:AbstractVector{<:AbstractFloat},
}
    δ = heaviside.(u .* rot)
    fWall = 0.5 .* δ .+ f .* (1.0 .- δ)
    @. ff = u * fWall * dt

    return nothing
end

maxwell_boltzmann(x) = x * log(x) - x

function kinetic_entropy(α, m, weights)
    B = KitBase.maxwell_boltzmann_dual_prime.(α' * m)[:]
    return sum(maxwell_boltzmann.(B) .* weights)
end

begin
    # setup
    set = Setup("radiation", "linesource", "1d1f1v", "kfvs", "bgk", 1, 2, "vanleer", "extra", 0.3, 0.8)

    # physical space
    x0 = 0
    x1 = 1
    nx = 100
    nxg = 0
    ps = PSpace1D(x0, x1, nx, nxg)

    # velocity space
    nu = 28
    points, weights = gausslegendre(nu)
    vs = VSpace1D(points[1], points[end], nu, points, ones(nu) .* (points[end] - points[1]) / (nu - 1), weights)

    # material
    σs = ones(Float32, nx)
    σa = zeros(Float32, nx)
    σt = σs + σa
    σq = zeros(Float32, nx)

    # moments
    L = 1
    ne = 2
    m = ComputeSphericalBasisAnalytical(points)

    # time
    cfl = 0.5
    dt = cfl * ps.dx[1]
    nt = set.maxTime / dt |> floor |> Int
    global t = 0.0

    # solution
    f0 = 0.0001 * ones(nu)
    phi = zeros(ne, nx)
    for i = 1:nx
        phi[:, i] .= m * f0
    end
    α = zeros(Float32, ne, nx)

    # NN
    αT = zeros(Float32, nx, ne)
    phiT = zeros(Float32, nx, ne)
    phi_old = zeros(Float32, ne, nx)
    phi_temp = deepcopy(phi_old)

    global X = zeros(Float32, ne, 1)
    #global Y = zeros(Float32, 1, 1) # h
    global Y = zeros(Float32, ne, 1) # α
    res = KitBase.optimize_closure(zeros(Float32, 2), m, weights, phi[:, 1], KitBase.maxwell_boltzmann_dual)
    X[:, 1] .= phi[:, 1]
    #Y[1, 1] = kinetic_entropy(res.minimizer, m, weights)
    Y[:, 1] = res.minimizer
end

begin
    # initial condition
    f0 = 0.0001 * ones(nu)
    phi = zeros(ne, nx)
    for i = 1:nx
        phi[:, i] .= m * f0
    end
    α = zeros(Float32, ne, nx)
    flux = zeros(Float32, ne, nx + 1)
    fη = zeros(nu)
end

#anim = @animate 
for iter = 1:nt
    println("iteration $iter of $nt")

    # mathematical optimizer
    @inbounds for i = 1:nx
        _res = KitBase.optimize_closure(α[:, i], m, weights, phi[:, i], KitBase.maxwell_boltzmann_dual)
        α[:, i] .= _res.minimizer
        phi[:, i] .= KitBase.realizable_reconstruct(_res.minimizer, m, weights, KitBase.maxwell_boltzmann_dual_prime)
    
        #X = hcat(X, phi[:, i])
        #Y = hcat(Y, kinetic_entropy(α[:, i], m, weights))
    end

    flux_wall!(fη, maxwell_boltzmann_dual.(α[:, 1]' * m)[:], points, dt, 1.0)
    for k in axes(flux, 1)
        flux[k, 1] = sum(m[k, :] .* weights .* fη)
    end

    @inbounds for i = 2:nx
        KitBase.flux_kfvs!(fη, KitBase.maxwell_boltzmann_dual.(α[:, i-1]' * m)[:], KitBase.maxwell_boltzmann_dual.(α[:, i]' * m)[:], points, dt)
        
        for k in axes(flux, 1)
            flux[k, i] = sum(m[k, :] .* weights .* fη)
        end
    end

    @inbounds for i = 1:nx-1
        for q = 1:1
            phi[q, i] =
                phi[q, i] +
                (flux[q, i] - flux[q, i+1]) / ps.dx[i] +
                (σs[i] * phi[q, i] - σt[i] * phi[q, i]) * dt +
                σq[i] * dt
        end

        for q = 2:ne
            phi[q, i] =
                phi[q, i] +
                (flux[q, i] - flux[q, i+1]) / ps.dx[i] +
                (-σt[i] * phi[q, i]) * dt
        end
    end
    phi[:, nx] .=  phi[:, nx-1]

    global t += dt

    #plot(ps.x[1:nx], phi[1, :])
end
#gif(anim, "newton1d.gif")

for i = 1:nx
    X = hcat(X, phi[:, i])
    #Y = hcat(Y, kinetic_entropy(α[:, i], m, weights))
    Y = hcat(Y, α[:, i])
end

X_old = deepcopy(X)
Y_old = deepcopy(Y)

fnn = FastChain(FastDense(2, 10, tanh), FastDense(10, 10, tanh), FastDense(10, 10, tanh), FastDense(10, 2))
res = KitML.sci_train(fnn, (X, Y); maxiters = 200)
res = KitML.sci_train(fnn, (X, Y), res.minimizer, LBFGS(); maxiters=10000)

#cd(@__DIR__)
#tmodel = KitML.load_model("best_model.h5"; mode = :tf)
#tmodel.fit(permutedims(X), permutedims(Y), epochs=10)
#tmodel.predict(permutedims(phi[:, 1]))
#KitML.neural_closure(tmodel, permutedims(phi[:, 1]))

#X = deepcopy(X_old)
#Y = deepcopy(Y_old)
begin
    # initial condition
    f0 = 0.0001 * ones(nu)
    phi = zeros(ne, nx)
    for i = 1:nx
        phi[:, i] .= m * f0
    end
    α = zeros(Float32, ne, nx)
    flux = zeros(Float32, ne, nx + 1)
    fη = zeros(nu)
end

anim = @animate for iter = 1:nt
#for iter = 1:1#nt
    println("iteration $iter of $nt")
    phi_old .= phi

    # regularization
    #α .= KitML.neural_closure(nn, res.minimizer, phi_old)
    α .= fnn(phi_old, res.minimizer)
    @inbounds Threads.@threads for i = 1:nx
        phi_temp[:, i] .= KitBase.realizable_reconstruct(α[:, i], m, weights, KitBase.maxwell_boltzmann_dual_prime)
    end

    counter = 0
    @inbounds for i = 1:nx
        if norm(phi_temp[:, i] .- phi_old[:, i], 1) / phi_old[1, i] > 2e-3
            counter +=1

            _res = KitBase.optimize_closure(α[:, i], m, weights, phi[:, i], KitBase.maxwell_boltzmann_dual)
            α[:, i] .= _res.minimizer
            phi[:, i] .= KitBase.realizable_reconstruct(_res.minimizer, m, weights, KitBase.maxwell_boltzmann_dual_prime)

            X = hcat(X, phi[:, i])
            #Y = hcat(Y, kinetic_entropy(α[:, i], m, weights))
            Y = hcat(Y, α[:, i])
        else
            phi[:, i] .= phi_temp[:, i]
        end
    end
    println("newton: $counter of $nx")

    flux_wall!(fη, maxwell_boltzmann_dual.(α[:, 1]' * m)[:], points, dt, 1.0)
    for k in axes(flux, 1)
        flux[k, 1] = sum(m[k, :] .* weights .* fη)
    end

    @inbounds for i = 2:nx
        KitBase.flux_kfvs!(fη, KitBase.maxwell_boltzmann_dual.(α[:, i-1]' * m)[:], KitBase.maxwell_boltzmann_dual.(α[:, i]' * m)[:], points, dt)
        
        for k in axes(flux, 1)
            flux[k, i] = sum(m[k, :] .* weights .* fη)
        end
    end

    @inbounds for i = 1:nx-1
        for q = 1:1
            phi[q, i] =
                phi[q, i] +
                (flux[q, i] - flux[q, i+1]) / ps.dx[i] +
                (σs[i] * phi[q, i] - σt[i] * phi[q, i]) * dt +
                σq[i] * dt
        end

        for q = 2:ne
            phi[q, i] =
                phi[q, i] +
                (flux[q, i] - flux[q, i+1]) / ps.dx[i] +
                (-σt[i] * phi[q, i]) * dt
        end
    end
    phi[:, nx] .=  phi[:, nx-1]

    global t += dt

    if iter%19 == 0
        res = KitML.sci_train(fnn, (X, Y), res.minimizer, LBFGS(); maxiters=500)
        
        #=X = zeros(Float32, 1, ne)
        Y = zeros(Float32, 1, 1)
        res = KitBase.optimize_closure(α[:, nx], m, weights, phi[:, nx], KitBase.maxwell_boltzmann_dual)
        X[1, :] .= phi[:, nx]
        Y[1, 1] = kinetic_entropy(α[:, nx], m, weights)=#
    end

    plot(ps.x[1:nx], phi[1, :])
end

gif(anim, "unified_1d.gif")
#gif(anim, "neural_1d.gif")
plot(ps.x[1:nx], phi[1, :])

using BenchmarkTools, Optim

@btime KitML.neural_closure(tmodel, phiT[:, :])
@btime for i = 1:nx
    #res = KitBase.optimize_closure(α[:, i], m, weights, phi[:, i], KitBase.maxwell_boltzmann_dual)

    loss(_α) = sum(maxwell_boltzmann_dual.(_α' * m)[:] .* weights) - dot(_α, phi[:, i])
    res = Optim.optimize(loss, α[:, i], Optim.Newton())
end

plot(Y)

