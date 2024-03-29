using LinearAlgebra, Plots, JLD2
using Flux, Flux.Zygote, Optim, DiffEqFlux
using KitBase, ProgressMeter
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
    nx = 60#100
    ny = 60#100
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
    init_field(x, y) = max(flr, 1.0 / (4.0 * pi * s2) * exp(-(x^2 + y^2) / 4.0 / s2))
    for j = 1:nx
        for i = 1:ny
            y = y0 + (i - 0.5) * dy
            x = x0 + (j - 0.5) * dx
            # only zeroth order moment is non-zero
            phi[1, i, j] = init_field(x, y)
        end
    end
end

global t = 0.0
flux1 = zeros(ne, nx + 1, ny)
flux2 = zeros(ne, nx, ny + 1)

# mechanical solver
anim = @animate for iter = 1:140
    println("iteration $(iter)")

    # regularization
    @inbounds Threads.@threads for j = 1:ny
        for i = 1:nx
            res = KitBase.optimize_closure(
                α[:, i, j],
                m,
                weights,
                phi[:, i, j],
                KitBase.maxwell_boltzmann_dual,
            )
            α[:, i, j] .= res.minimizer

            phi[:, i, j] .= KitBase.realizable_reconstruct(
                res.minimizer,
                m,
                weights,
                KitBase.maxwell_boltzmann_dual_prime,
            )
        end
    end

    # flux
    fη1 = zeros(nq)
    @inbounds for j = 1:ny
        for i = 2:nx
            KitBase.flux_kfvs!(
                fη1,
                KitBase.maxwell_boltzmann_dual.(α[:, i-1, j]' * m)[:],
                KitBase.maxwell_boltzmann_dual.(α[:, i, j]' * m)[:],
                points[:, 1],
                dt,
            )

            for k in axes(flux1, 1)
                flux1[k, i, j] = sum(m[k, :] .* weights .* fη1)
            end
        end
    end

    fη2 = zeros(nq)
    @inbounds for i = 1:nx
        for j = 2:ny
            KitBase.flux_kfvs!(
                fη2,
                KitBase.maxwell_boltzmann_dual.(α[:, i, j-1]' * m)[:],
                KitBase.maxwell_boltzmann_dual.(α[:, i, j]' * m)[:],
                points[:, 2],
                dt,
            )

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
                    (SigmaS[i, j] * phi[q, i, j] - SigmaT[i, j] * phi[q, i, j]) * dt
            end

            for q = 2:ne
                phi[q, i, j] =
                    phi[q, i, j] +
                    (flux1[q, i, j] - flux1[q, i+1, j]) / dx +
                    (flux2[q, i, j] - flux2[q, i, j+1]) / dy +
                    (-SigmaT[i, j] * phi[q, i, j]) * dt
            end
        end
    end

    global t += dt
    contourf(pspace.x[1:nx, 1], pspace.y[1, 1:ny], phi[1, :, :])
end

cd(@__DIR__)
gif(anim, "linesource_mn.gif")

begin
    # create neural network
    #nn = KitML.create_neural_closure(ne, 1)
    nn = KitML.ICNNChain(4, 1, [10, 10, 10], tanh)

    # load saved params
    #cd(@__DIR__)
    #@load "model.jld2" nn
    #@info "model loaded"
end

X = zeros(4, 1)
Y = zeros(1, 1)
X[:, 1] .= phi[:, 1, 1]
Y[1, 1] =
    sum(weights .* maxwell_boltzmann_dual.(maxwell_boltzmann_dual_prime.(m' * α[:, 1, 1])))

for i = 1:nx, j = 1:ny
    X = hcat(X, phi[:, i, j])

    η = sum(
        weights .* maxwell_boltzmann_dual.(maxwell_boltzmann_dual_prime.(m' * α[:, i, j])),
    )
    Y = hcat(Y, η)
end



X = zeros(4, 1)
Y = zeros(4, 1)
X[:, 1] .= phi[:, 1, 1]
Y[:, 1] .= α[:, 1, 1]

for i = 21:30, j = 21:30
    X = hcat(X, phi[:, i, j])
    Y = hcat(Y, α[:, i, j])
end


nn = Chain(Dense(4, 16, tanh), Dense(16, 16, tanh), Dense(16, 16, tanh), Dense(16, 4))
data = Flux.Data.DataLoader(X, Y, batchsize = 50, shuffle = true)
KitML.sci_train!(nn, data, ADAM(), 500)

# unified solver
@showprogress for iter = 1:5
    #=
    # regularization
    Threads.@threads for j = 1:ny
        for i = 1:nx
            phi_old = phi


            #res = KitBase.optimize_closure(α[:, i, j], m, weights, phi[:, i, j], KitBase.maxwell_boltzmann_dual)
            #α[:, i, j] .= res.minimizer
            #phi[:, i, j] .= KitBase.realizable_reconstruct(res.minimizer, m, weights, KitBase.maxwell_boltzmann_dual_prime)

            #training phase network
            KitML.sci_train!(nn, (phi[:, i, j], α[:, i, j]))

            # calculate η' = f'(u0, u1, u2, u3)
            α[:, i, j] = gradient(x -> first(nn(x)), phi[:, i, j])[1]
            phi[:, i, j] .= KitBase.realizable_reconstruct(
                α[:, i, j],
                m,
                weights,
                KitBase.maxwell_boltzmann_dual_prime,
            )
        end
    end

    KitML.sci_train!(nn, (phi[:, i, j], η[i, j]))
    =#
    #=
    Threads.@threads for j = 1:ny
        for i = 1:nx
            α[:, i, j] .= nn(phi[:, i, j])
            phi[:, i, j] .= KitBase.realizable_reconstruct(
                α[:, i, j],
                m,
                weights,
                KitBase.maxwell_boltzmann_dual_prime,
            )
        end
    end
    =#

    # flux
    fη1 = zeros(nq)
    for j = 1:ny
        for i = 2:nx
            #αL = gradient(x -> first(nn(x)), phi[:, i-1, j])[1]
            #αR = gradient(x -> first(nn(x)), phi[:, i, j])[1]
            αL = nn(phi[:, i-1, j])
            αR = nn(phi[:, i, j])
            KitBase.flux_kfvs!(
                fη1,
                KitBase.maxwell_boltzmann_dual.(αL' * m)[:],
                KitBase.maxwell_boltzmann_dual.(αR' * m)[:],
                points[:, 1],
                dt,
            )

            for k in axes(flux1, 1)
                flux1[k, i, j] = sum(m[k, :] .* weights .* fη1)
            end
        end
    end

    fη2 = zeros(nq)
    for i = 1:nx
        for j = 2:ny
            #αL = gradient(x -> first(nn(x)), phi[:, i, j-1])[1]
            #αR = gradient(x -> first(nn(x)), phi[:, i, j])[1]
            αL = nn(phi[:, i, j-1])
            αR = nn(phi[:, i, j])
            KitBase.flux_kfvs!(
                fη2,
                KitBase.maxwell_boltzmann_dual.(αL' * m)[:],
                KitBase.maxwell_boltzmann_dual.(αR' * m)[:],
                points[:, 2],
                dt,
            )

            for k in axes(flux2, 1)
                flux2[k, i, j] = sum(m[k, :] .* (weights .* fη2))
            end
        end
    end

    # update
    for j = 2:ny-1
        for i = 2:nx-1
            for q = 1:1
                phi[q, i, j] =
                    phi[q, i, j] +
                    (flux1[q, i, j] - flux1[q, i+1, j]) / dx +
                    (flux2[q, i, j] - flux2[q, i, j+1]) / dy +
                    (SigmaS[i, j] * phi[q, i, j] - SigmaT[i, j] * phi[q, i, j]) * dt
            end

            for q = 2:ne
                phi[q, i, j] =
                    phi[q, i, j] +
                    (flux1[q, i, j] - flux1[q, i+1, j]) / dx +
                    (flux2[q, i, j] - flux2[q, i, j+1]) / dy +
                    (-SigmaT[i, j] * phi[q, i, j]) * dt
            end
        end
    end

    global t += dt
end

using Plots
contourf(pspace.x[1:nx, 1], pspace.y[1, 1:ny], phi[1, :, :])
#contourf(pspace.x[1:nx, 1], pspace.y[1, 1:ny], α[1, :, :])

# saving neural network Progress
@save "model.jld2" nn
@info "model saved"

res = KitBase.optimize_closure(
    α[:, 25, 25],
    m,
    weights,
    phi[:, 25, 25],
    KitBase.maxwell_boltzmann_dual,
)
res.minimizer

savefig("test.png")
