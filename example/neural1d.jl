using LinearAlgebra, Plots, JLD2
using Flux, Flux.Zygote, Optim, DiffEqFlux
using KitBase, ProgressMeter
import KitML

maxwell_boltzmann(x) = x * log(x) - x

function kinetic_entropy(α, m, weights)
    B = KitBase.maxwell_boltzmann_dual_prime.(α' * m)[:]
    return sum(maxwell_boltzmann.(B) .* weights)
end

cd(@__DIR__)
include("math.jl")
model = KitML.load_model("best_model.h5"; mode = :tf)

begin
    # space
    x0 = 0.0
    x1 = 1.0
    nx = 100
    dx = (x1 - x0) / nx

    pspace = KitBase.PSpace1D(x0, x1, nx)

    # time
    tEnd = 1.0
    cfl = 0.5
    dt = cfl * dx

    # quadrature
    quadratureorder = 5
    points, triangulation = KitBase.octa_quadrature(quadratureorder)
    weights = KitBase.quadrature_weights(points, triangulation)
    nq = size(points, 1)

    # moments
    L = 1
    ne = (L + 1)^2
    #m = KitBase.eval_spherharmonic(points, L)
    m = ComputeSphericalBasisAnalytical(L, 3, points)

    # collision
    σs = ones(Float32, nx)
    σa = zeros(Float32, nx)# * 0.1
    #σa = exp.(pspace.x[1:nx])

    σt = σs + σa
    σq = zeros(Float32, nx)
    #σq = 1e-10 .* exp.(-100 * pspace.x[1:nx])

    global t = 0.0
    flux = zeros(Float32, ne, nx + 1)
end

pdf = zeros(Float32, axes(weights))
begin
    α = zeros(Float32, ne, nx)
    phi = ones(Float32, ne, nx) .* 1e-6
    phi[1, 1] = 0.5
    #=
    for i = 1:nx
        #=for j = 1:nq
            if points[j, 1] > 0.
                pdf[j] = (1.0 + sin(2π * pspace.x[i])) ./ 4π
            end
        end
        phi[:, i] .= m * pdf=#

        #phi[1, i] = 1.0 #- 0.05 * (pspace.x[i] - x0)
        #phi[1, i] = 1.0 + 0.2 * sin(2π * pspace.x[i])
        #phi[2, i] = 0.2
    end=#
end

#αT = zeros(Float32, nx, ne)
#phiT = zeros(Float32, nx, ne)

#global X = zeros(Float32, 1, ne)
#global Y = zeros(Float32, 1, 1)

#res = KitBase.optimize_closure(X[1, :], m, weights, phi[:, nx÷2], KitBase.maxwell_boltzmann_dual)
#X[1, :] .= res.minimizer       
#Y[1, 1] = kinetic_entropy(X[1, :], m, weights)

@showprogress for iter = 1:20#20
    #phi_old .= phi
    #=
        # regularization
        @inbounds for i = 1:ne
            phiT[:, i] .= phi[i, :, :][:]
        end
        αT .= KitML.neural_closure(model, phiT)
        @inbounds for i = 1:ne
            α[i, :, :] .= reshape(αT[:, i], nx, ny)
        end
        @inbounds Threads.@threads for j = 1:ny
            for i = 1:nx
                phi_temp[:, i, j] .= KitBase.realizable_reconstruct(α[:, i, j], m, weights, KitBase.maxwell_boltzmann_dual_prime)
            end
        end

        @inbounds for j = 1:ny
            for i = 1:nx
                #@show norm(phi_temp[:, i, j] - phi_old[:, i, j], 2)

                if norm(phi_temp[:, i, j] - phi_old[:, i, j], 2) > 1e-1
                    res = KitBase.optimize_closure(α[:, i, j], m, weights, phi[:, i, j], KitBase.maxwell_boltzmann_dual)
                    α[:, i, j] .= res.minimizer
                    phi[:, i, j] .= KitBase.realizable_reconstruct(res.minimizer, m, weights, KitBase.maxwell_boltzmann_dual_prime)

                    if phi[1, i, j] > 0.01
                        X = vcat(X, permutedims(α[:, i, j]))
                        Y = vcat(Y, kinetic_entropy(α[:, i, j], m, weights))
                    end
                else
                    phi[:, i, j] .= phi_temp[:, i, j]
                end
            end
        end
    =#

    res = KitBase.optimize_closure(
        α[:, 1],
        m,
        weights,
        phi[:, 1],
        KitBase.maxwell_boltzmann_dual,
    )
    α[:, 1] .= res.minimizer
    #res = KitBase.optimize_closure(α[:, nx], m, weights, phi[:, nx], KitBase.maxwell_boltzmann_dual)
    #α[:, nx] .= res.minimizer

    @inbounds for i = 2:nx
        res = KitBase.optimize_closure(
            α[:, i],
            m,
            weights,
            phi[:, i],
            KitBase.maxwell_boltzmann_dual;
            optimizer = Newton(),
        )
        α[:, i] .= res.minimizer
        phi[:, i] .= KitBase.realizable_reconstruct(
            res.minimizer,
            m,
            weights,
            KitBase.maxwell_boltzmann_dual_prime,
        )
    end

    # flux
    fη = zeros(nq)
    @inbounds for i = 2:nx
        KitBase.flux_kfvs!(
            fη,
            KitBase.maxwell_boltzmann_dual.(α[:, i-1]' * m)[:],
            KitBase.maxwell_boltzmann_dual.(α[:, i]' * m)[:],
            points[:, 1],
            dt,
        )

        for k in axes(flux, 1)
            flux[k, i] = sum(m[k, :] .* weights .* fη)
        end
    end
    #=
    # periodic
    KitBase.flux_kfvs!(fη, KitBase.maxwell_boltzmann_dual.(α[:, nx]' * m)[:], KitBase.maxwell_boltzmann_dual.(α[:, 1]' * m)[:], points[:, 1], dt)
    for k in axes(flux, 1)
        flux[k, 1] = sum(m[k, :] .* weights .* fη)
        flux[k, nx+1] = flux[k, 1]
    end
    =#

    # update
    @inbounds for i = 2:nx-1
        for q = 1:1
            phi[q, i] =
                phi[q, i] +
                (flux[q, i] - flux[q, i+1]) / dx +
                (σs[i] * phi[q, i] - σt[i] * phi[q, i]) * dt +
                σq[i] * dt
        end

        for q = 2:ne
            phi[q, i] =
                phi[q, i] + (flux[q, i] - flux[q, i+1]) / dx + (-σt[i] * phi[q, i]) * dt
        end
    end

    phi[:, nx] .= phi[:, nx-1]

    global t += dt

    #=
    if iter%9 == 0
        model.fit(X, Y, epochs=2)

        X = zeros(Float32, 1, ne)
        Y = zeros(Float32, 1, 1)
        res = KitBase.optimize_closure(X[1, :], m, weights, phi[:, nx÷2, ny÷2], KitBase.maxwell_boltzmann_dual)
        X[1, :] .= res.minimizer       
        Y[1, 1] = kinetic_entropy(X[1, :], m, weights)
    end=#
end

scatter(pspace.x[1:nx], phi[1, :])
scatter(pspace.x[1:nx], α[1, :])
