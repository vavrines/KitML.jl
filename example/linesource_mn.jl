using ProgressMeter, LinearAlgebra, Optim, JLD2, KitBase
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
    u0 = [2., 0., 0., 0.]
    m = KitBase.eval_spherharmonic(points, L)

    res = KitBase.optimize_closure(α, m, weights, u0, KitBase.maxwell_boltzmann_dual)
    u = KitBase.realizable_reconstruct(res.minimizer, m, weights, KitBase.maxwell_boltzmann_dual_prime)
end

# multi-cell case
begin
    # space
    x0 = -1.5
    x1 = 1.5
    y0 = -1.5
    y1 = 1.5
    nx = 50#100
    ny = 50#100
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

begin
    # create neural network
    global nn_model = KitML.create_neural_closure(ne, ne)

    # load saved params
    #@load "nn_model.jld2" nn_model
    println("model loaded")
end 

global t = 0.0
flux1 = zeros(ne, nx + 1, ny)
flux2 = zeros(ne, nx, ny + 1)

#@showprogress for iter = 1:10
    # regularization
    Threads.@threads for j = 1:1#ny
        for i = 1:1#nx
            res = KitBase.optimize_closure(α[:, i, j], m, weights, phi[:, i, j], KitBase.maxwell_boltzmann_dual)
            α[:, i, j] .= res.minimizer
            
            phi[:, i, j] .= KitBase.realizable_reconstruct(res.minimizer, m, weights, KitBase.maxwell_boltzmann_dual_prime)
            
            #training phase network
            KitML.sci_train!(nn_model, (phi[:, i, j], α[:, i, j]))

            #Execution phase
            #α[:, i, j]=nn_model(phi[:, i, j])
            #phi[:, i, j] .= KitBase.realizable_reconstruct(res.minimizer, m, weights, KitBase.maxwell_boltzmann_dual_prime)
        end
    end
    
    # flux
    fη1 = zeros(nq)
    for j = 1:ny
        for i = 2:nx
            KitBase.flux_kfvs!(fη1, KitBase.maxwell_boltzmann_dual.(α[:, i-1, j]' * m)[:], KitBase.maxwell_boltzmann_dual.(α[:, i, j]' * m)[:], points[:, 1], dt)
            
            for k in axes(flux1, 1)
                flux1[k, i, j] = sum(m[k, :] .* weights .* fη1)
            end
        end
    end

    fη2 = zeros(nq)
    for i = 1:nx
        for j = 2:ny
            KitBase.flux_kfvs!(fη2, KitBase.maxwell_boltzmann_dual.(α[:, i, j-1]' * m)[:], KitBase.maxwell_boltzmann_dual.(α[:, i, j]' * m)[:], points[:, 2], dt)
            
            for k in axes(flux2, 1)
                flux2[k, i, j] = sum(m[k, :] .* (weights .* fη2))
            end
        end
    end
    
    # update
    for j = 2:ny-1
        for i = 2:nx-1
            for q = 1:ne
                phi[q, i, j] =
                    phi[q, i, j] +
                    (flux1[q, i, j] - flux1[q, i+1, j]) / dx +
                    (flux2[q, i, j] - flux2[q, i, j+1]) / dy #+
                    #(integral - phi[q, i, j]) * dt
            end
        end
    end

    global t += dt
#end

# saving neural network Progress
@save "nn_model.jld2" nn_model
println("model saved")

using Plots
contourf(pspace.x[1:nx, 1], pspace.y[1, 1:ny], α[1, :, :])
contourf(pspace.x[1:nx, 1], pspace.y[1, 1:ny], phi[1, :, :])
