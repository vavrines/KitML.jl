using LinearAlgebra, Plots, JLD2
using Flux, Flux.Zygote, Optim, DiffEqFlux
using KitBase
import KitML

# one-cell simplification
begin
    quadratureorder = 5
    points, triangulation = KitBase.octa_quadrature(quadratureorder)
    weights = KitBase.quadrature_weights(points, triangulation)
    nq = size(points, 1)
    L = 1
    ne = (L + 1)^2

    α = zeros(ne)
    u0 = [2.0, 0.0, 0.0, 0.0]
    m = KitBase.eval_spherharmonic(points, L)

    res = KitBase.optimize_closure(α, m, weights, u0, KitBase.maxwell_boltzmann_dual)
    u = KitBase.realizable_reconstruct(
        res.minimizer,
        m,
        weights,
        KitBase.maxwell_boltzmann_dual_prime,
    )
end

# multi-cell case
begin
    # space
    x0 = -1.5
    x1 = 1.5
    y0 = -1.5
    y1 = 1.5
    nx = 100
    ny = 100
    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny

    pspace = KitBase.PSpace2D(x0, x1, nx, y0, y1, ny)

    # time
    tEnd = 1.0
    cfl = 0.8
    dt = cfl / 2 * (dx * dy) / (dx + dy)

    # quadrature
    quadratureorder = 5
    points, triangulation = KitBase.octa_quadrature(quadratureorder)
    weights = KitBase.quadrature_weights(points, triangulation)
    nq = size(points, 1)

    # particle
    SigmaS = 1 * ones(ny + 4, nx + 4)
    SigmaA = 0 * ones(ny + 4, nx + 4)
    SigmaT = SigmaS + SigmaA

    # moments
    L = 1
    ne = (L + 1)^2
    phi = zeros(ne, nx, ny)
    α = zeros(ne, nx, ny)
    m = KitBase.eval_spherharmonic(points, L)
end

# initial distribution
begin
    s2 = 0.03^2
    flr = 1e-4
    init_field(x, y) = (1.5+cos(2*π*x)*cos(2*π*y))
    for j = 1:nx
        for i = 1:ny
            y = y0 + (i - 0.5) * dy
            x = x0 + (j - 0.5) * dx
            # only zeroth order moment is non-zero
            phi[1, i, j] = init_field(x, y)
            phi[2, i, j] = 0.9/3.0 * phi[1, i, j]
            phi[3, i, j] = 0.9/3.0 * phi[1, i, j]
            phi[4, i, j] = 0.9/3.0 * phi[1, i, j]
        end
    end
end

global t = 0.0
flux1 = zeros(ne, nx + 1, ny)
flux2 = zeros(ne, nx, ny + 1)

begin
# mechanical solver
anim = @animate for iter = 1:200
    println("iteration $(iter)")

    # regularization
    @inbounds Threads.@threads for j = 1:ny
        for i = 1:nx
            res = KitBase.optimize_closure(α[:, i, j], m, weights, phi[:, i, j], KitBase.maxwell_boltzmann_dual)
            α[:, i, j] .= res.minimizer
            
            phi[:, i, j] .= KitBase.realizable_reconstruct(res.minimizer, m, weights, KitBase.maxwell_boltzmann_dual_prime)
        end
    end
    
    # flux
    fη1 = zeros(nq)
    @inbounds for j = 1:ny
        for i = 1:nx
            im1 = i-1
            if i == 1  #periodic boundaries
                im1 = nx
            end

            KitBase.flux_kfvs!(fη1, KitBase.maxwell_boltzmann_dual.(α[:, im1, j]' * m)[:], KitBase.maxwell_boltzmann_dual.(α[:, i, j]' * m)[:], points[:, 1], dt)
            
            for k in axes(flux1, 1)
                flux1[k, i, j] = sum(m[k, :] .* weights .* fη1)
            end
        end
    end

    fη2 = zeros(nq)
    @inbounds for i = 1:nx
        for j = 1:ny
            jm1 = j-1
            if j == 1  #periodic boundaries
                jm1 = ny
            end
            KitBase.flux_kfvs!(fη2, KitBase.maxwell_boltzmann_dual.(α[:, i, jm1]' * m)[:], KitBase.maxwell_boltzmann_dual.(α[:, i, j]' * m)[:], points[:, 2], dt)
            
            for k in axes(flux2, 1)
                flux2[k, i, j] = sum(m[k, :] .* (weights .* fη2))
            end
        end
    end
    
    # update
    @inbounds for j = 1:ny
        for i = 1:nx
            ip1 = i + 1
            jp1 = j + 1
            
            #periodic boundaries
            if i==nx
                ip1 = 1
            end
            if j==ny
                jp1 = 1
            end

            for q = 1:1
                phi[q, i, j] =
                    phi[q, i, j] +
                    (flux1[q, i, j] - flux1[q, ip1, j]) / dx +
                    (flux2[q, i, j] - flux2[q, i, jp1]) / dy +
                    (SigmaS[i, j] * phi[q, i, j] - SigmaT[i, j] * phi[q, i, j]) * dt
            end

            for q = 2:ne
                phi[q, i, j] =
                    phi[q, i, j] +
                    (flux1[q, i, j] - flux1[q, ip1, j]) / dx +
                    (flux2[q, i, j] - flux2[q, i, jp1]) / dy +
                    (- SigmaT[i, j] * phi[q, i, j]) * dt
            end
        end
    end

    global t += dt
    contourf(pspace.x[1:nx, 1], pspace.y[1, 1:ny], phi[1, :, :], clims=(0, 3))
end
end
cd(@__DIR__)
gif(anim, "periodic_mn.gif")