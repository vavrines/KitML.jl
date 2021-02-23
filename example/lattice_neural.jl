using LinearAlgebra, Plots, JLD2
using Flux, Flux.Zygote, Optim, DiffEqFlux
using KitBase, ProgressMeter
import KitML

maxwell_boltzmann(x) = x * log(x) - x

maxwell_boltzmann_prime(x) = log(x)

function kinetic_entropy(α, m, weights)
    B = KitBase.maxwell_boltzmann_dual_prime.(α' * m)[:]
    return sum(maxwell_boltzmann.(B) .* weights)
end

function is_absorb(x::T, y::T) where {T<:Real}
    cds = Array{Bool}(undef, 11) # conditions

    cds[1] = -2.5<x<-1.5 && 1.5<y<2.5
    cds[2] = -2.5<x<-1.5 && -0.5<y<0.5
    cds[3] = -2.5<x<-1.5 && -2.5<y<-1.5
    cds[4] = -1.5<x<-0.5 && 0.5<y<1.5
    cds[5] = -1.5<x<-0.5 && -1.5<y<-0.5
    cds[6] = -0.5<x<0.5 && -2.5<y<-1.5
    cds[7] = 0.5<x<1.5 && 0.5<y<1.5
    cds[8] = 0.5<x<1.5 && -1.5<y<-0.5
    cds[9] = 1.5<x<2.5 && 1.5<y<2.5
    cds[10] = 1.5<x<2.5 && -0.5<y<0.5
    cds[11] = 1.5<x<2.5 && -2.5<y<-1.5

    if any(cds) == true
        return true
    else
        return false
    end
end

cd(@__DIR__)
include("math.jl")
model = KitML.load_model("best_model.h5"; mode = :tf)

begin
    # space
    x0 = -3.5
    x1 = 3.5
    y0 = -3.5
    y1 = 3.5
    nx = 80#100
    ny = 80#100
    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny

    pspace = KitBase.PSpace2D(x0, x1, nx, y0, y1, ny)

    # time
    tEnd = 1.0
    cfl = 0.5
    dt = cfl / 2 * (dx * dy) / (dx + dy)

    # quadrature
    quadratureorder = 5
    points, triangulation = KitBase.octa_quadrature(quadratureorder)
    weights = KitBase.quadrature_weights(points, triangulation)
    nq = size(points, 1)

    # moments
    L = 1
    ne = (L + 1)^2
    phi = zeros(Float32, ne, nx, ny)
    phi[1, :, :] .= 1e-6
    α = zeros(Float32, ne, nx, ny)
    #m = KitBase.eval_spherharmonic(points, L)
    m = ComputeSphericalBasisAnalytical(L, 3, points)
end

begin
    σs = zeros(Float64, nx, ny)
    σa = zeros(Float64, nx, ny)
    for i = 1:nx, j = 1:ny
        if is_absorb(pspace.x[i, j], pspace.y[i, j])
            σs[i, j] = 0.0
            σa[i, j] = 10.0
        else
            σs[i, j] = 1.0
            σa[i, j] = 0.0
        end
    end
    σt = σs + σa
    σq = zeros(Float64, nx, ny)
    for i = 1:nx, j = 1:ny
        if -0.5<pspace.x[i, j]<0.5 && -0.5<pspace.y[i, j]<0.5
            σq[i, j] = 1.0 / (4.0 * π)
        else
            σq[i, j] = 0.0
        end
    end
end

#contourf(pspace.x[1:nx, 1], pspace.y[1, 1:ny], σs')

global t = 0.0
flux1 = zeros(Float32, ne, nx + 1, ny)
flux2 = zeros(Float32, ne, nx, ny + 1)

αT = zeros(Float32, nx*ny, ne)
phiT = zeros(Float32, nx*ny, ne)

global X = zeros(Float32, 1, ne)
global Y = zeros(Float32, 1, 1)

res = KitBase.optimize_closure(X[1, :], m, weights, phi[:, nx÷2, ny÷2], KitBase.maxwell_boltzmann_dual)
X[1, :] .= res.minimizer       
Y[1, 1] = kinetic_entropy(X[1, :], m, weights)

#=
# compare mathematical & NN optimizer
α0 = similar(α)
@inbounds Threads.@threads for j = 1:ny
    for i = 1:nx
        res = KitBase.optimize_closure(α[:, i, j], m, weights, phi[:, i, j], KitBase.maxwell_boltzmann_dual)
        α0[:, i, j] .= res.minimizer
    end
end

@inbounds for i = 1:ne
    phiT[:, i] .= phi[i, :, :][:]
end
αT = KitML.neural_closure(model, phiT)
α1 = similar(α)
@inbounds for i = 1:ne
    α1[i, :, :] .= reshape(αT[:, i], nx, ny)
end

plot(pspace.x[1:nx, 1], α0[1, :, 40], label="Newton", xlabel="x", ylabel="alpha_0")
plot!(pspace.x[1:nx, 1], α1[1, :, 40], label="NN")
=#

phi_temp = zero(phi)
phi_old = zero(phi)
@showprogress for iter = 1:10#20
    phi_old .= phi

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

    # flux
    fη1 = zeros(nq)
    @inbounds for j = 1:ny
        for i = 2:nx
            KitBase.flux_kfvs!(fη1, KitBase.maxwell_boltzmann_dual.(α[:, i-1, j]' * m)[:], KitBase.maxwell_boltzmann_dual.(α[:, i, j]' * m)[:], points[:, 1], dt)
            
            for k in axes(flux1, 1)
                flux1[k, i, j] = sum(m[k, :] .* weights .* fη1)
            end
        end
    end

    fη2 = zeros(nq)
    @inbounds for i = 1:nx
        for j = 2:ny
            KitBase.flux_kfvs!(fη2, KitBase.maxwell_boltzmann_dual.(α[:, i, j-1]' * m)[:], KitBase.maxwell_boltzmann_dual.(α[:, i, j]' * m)[:], points[:, 2], dt)
            
            for k in axes(flux2, 1)
                flux2[k, i, j] = sum(m[k, :] .* (weights .* fη2))
            end
        end
    end
    
    # update
    @inbounds for j = 2:ny-1
        for i = 2:nx-1
            for q = 1:1
                phi[q, i, j] =
                    phi[q, i, j] +
                    (flux1[q, i, j] - flux1[q, i+1, j]) / dx +
                    (flux2[q, i, j] - flux2[q, i, j+1]) / dy +
                    (σs[i, j] * phi[q, i, j] - σt[i, j] * phi[q, i, j]) * dt +
                    σq[i, j] * dt * 100.0
            end

            for q = 2:ne
                phi[q, i, j] =
                    phi[q, i, j] +
                    (flux1[q, i, j] - flux1[q, i+1, j]) / dx +
                    (flux2[q, i, j] - flux2[q, i, j+1]) / dy +
                    (-σt[i, j] * phi[q, i, j]) * dt
            end
        end
    end

    global t += dt

    if iter%9 == 0
        model.fit(X, Y, epochs=2)
        
        X = zeros(Float32, 1, ne)
        Y = zeros(Float32, 1, 1)
        res = KitBase.optimize_closure(X[1, :], m, weights, phi[:, nx÷2, ny÷2], KitBase.maxwell_boltzmann_dual)
        X[1, :] .= res.minimizer       
        Y[1, 1] = kinetic_entropy(X[1, :], m, weights)
    end
end

contourf(pspace.x[1:nx, 1], pspace.y[1, 1:ny], phi[1, :, :])