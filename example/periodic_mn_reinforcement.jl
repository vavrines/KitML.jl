
using KitBase
using LinearAlgebra, Plots, JLD2
using Optim, DiffEqFlux
using CSV,DataFrames
using PyCall

import KitML
cd(@__DIR__)
include("math.jl")

# model initialization
# Force CPU Learning
py"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
"""

model = KitML.load_model("best_model.h5"; mode = :tf)

begin
    #single cell case
end
# Variable initialization
begin
    # space
    x0 = -1.5
    x1 = 1.5
    y0 = -1.5
    y1 = 1.5
    nx = 100#100
    ny = 100#100
    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny

    pspace = KitBase.PSpace2D(x0, x1, nx, y0, y1, ny)

    # time
    tEnd = 1.0
    cfl = 0.8
    dt = cfl / 2 * (dx * dy) / (dx + dy)

    # quadrature
    quadratureorder = 20
    points, triangulation = KitBase.octa_quadrature(quadratureorder)
    weights = KitBase.quadrature_weights(points, triangulation)
    nq = size(points, 1)

    # particle
    SigmaS = 0 * ones(ny + 4, nx + 4)
    SigmaA = 0 * ones(ny + 4, nx + 4)
    SigmaT = SigmaS + SigmaA

    # moments
    L = 1
    ne = GetBasisSize(L,3)
    phi = zeros(Float32, ne, nx, ny)
    α = zeros(Float32, ne, nx, ny)
    m = ComputeSphericalBasisKarth(points,1,3)
end



global t = 0.0
flux1 = zeros(ne, nx + 1, ny)
flux2 = zeros(ne, nx, ny + 1)

αT = zeros(nx*ny, ne)
phiT = zeros(nx*ny, ne)



#metaIter = 2
for metaIter in 1:500
# csv logging
df = DataFrame(Iter = Int64[], RelErrCells = Float64[], MaxRelError = Float64[])

# initial distribution
begin
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

println("metaIteration $(metaIter)")

# mechanical solver

for iter in 1:200
    println("iteration $(iter) , meta-iter $(metaIter)")

    #restructuring of the matrix
    @inbounds for i = 1:ne
        phiT[:, i] .= phi[i, :, :][:]
    end
    #Call the neural network
    αT = KitML.neural_closure(model, phiT)
    @inbounds for i = 1:ne
        α[i, :, :] .= reshape(αT[:, i], nx, ny)
    end

    #### Introduce helper variables
    phi_old = copy(phi)
    phi_train = Vector()
    h_train = Vector()
    errorsFound = false
    #####

    # realizability reconstruction
    error = 0.0
    count = 0
    maxError = 0.0
    @inbounds for j = 1:ny
        for i = 1:nx
            phi[:, i, j] .= KitBase.realizable_reconstruct(α[:, i, j], m, weights, KitBase.maxwell_boltzmann_dual_prime)
            # Check the prediction error
            error = norm( (phi[:, i, j]-phi_old[:, i, j] )/phi_old[:, i, j] ,2)^2 # take care of the criterion
            if error > maxError
                maxError = error
            end
            #println(error)
            if error > 5e-4 || phi[:, i, j] == NaN
                #println(error)
                errorsFound = true
                count = count +1
            end            
        end
    end

    println("Max error: $(maxError)")
    #retrain network on cells with too much error, if there is any
    if errorsFound
        for j = 1:ny
            for i = 1:nx
                #add point to retraining batch. compute alpha and h w.r.t phi_old
                res = KitBase.optimize_closure(α[:, i, j], m, weights, phi_old[:, i, j], KitBase.maxwell_boltzmann_dual)
                α[:, i, j] .= res.minimizer
                phi[:, i, j] .= KitBase.realizable_reconstruct(α[:, i, j], m, weights, KitBase.maxwell_boltzmann_dual_prime)

                #Setup trianing data
                h = computeEntropyH(α[:, i, j], m, weights,maxwell_boltzmann_primal, KitBase.maxwell_boltzmann_dual_prime)
                h_train =vcat(h_train, h)
                phi_train =vcat(phi_train, phi[:, i, j])
            end
        end

        print("Current amount of  errorous datapoints: ")
        print(size(h_train)[1]/(nx*ny)*100)
        print(" % ")
        print(" in iter ")
        println(iter)

        # Transform into matrix format
        phiTrainMat = zeros(size(h_train)[1],4)
        hTrainMat = zeros(size(h_train)[1],1)
        for i in 1:size(h_train)[1]
            phiTrainMat[i,1] = phi_train[4*(i-1)+1]
            phiTrainMat[i,2] = phi_train[4*(i-1)+2]
            phiTrainMat[i,3] = phi_train[4*(i-1)+3]
            phiTrainMat[i,4] = phi_train[4*(i-1)+4]
            hTrainMat[i,1]= h_train[i]
        end

        print(size(h_train))
        model.fit(phiTrainMat,hTrainMat,epochs =200,batch_size = 32,validation_split=0.0,verbose= 1)
        println("model retrained")
        model.save("best_model.h5") 
    else
        println("Error smaller than 5e-4")
    end
    
    push!(df,[iter , count/(nx*ny)*100, maxError])
    
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
    #contourf(pspace.x[1:nx, 1], pspace.y[1, 1:ny], phi[1, :, :], clims=(0.5, 2.5))
end

    CSV.write("Hist_metaIter_$(metaIter).csv",df)
end

#contourf(pspace.x[1:nx, 1], pspace.y[1, 1:ny], phi[1, :, :])
#cd(@__DIR__)
#gif(anim, "periodic_reinforcment_$(metaIter).gif")
