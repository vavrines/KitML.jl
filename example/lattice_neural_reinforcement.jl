using LinearAlgebra, Plots
using KitBase, ProgressMeter
import KitML

cd(@__DIR__)
include("math.jl")


model = KitML.load_model("best_model.h5"; mode = :tf)

begin
    # space
    x0 = -3.5
    x1 = 3.5
    y0 = -3.5
    y1 = 3.5
    nx = 50#100
    ny = 50#100
    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny

    pspace = KitBase.PSpace2D(x0, x1, nx, y0, y1, ny)

    # time
    tEnd = 5.0
    cfl = 0.7
    dt = cfl / 2 * (dx * dy) / (dx + dy)

    # quadrature
    quadratureorder = 5
    points, triangulation = KitBase.octa_quadrature(quadratureorder)
    weights = KitBase.quadrature_weights(points, triangulation)
    nq = size(points, 1)

    # moments
    L = 1
    #ne = (L + 1)^2
    ne = GetBasisSize(L,3)

    phi = zeros(Float32, ne, nx, ny)
    α = zeros(Float32, ne, nx, ny)
    h = 
    #m = KitBase.eval_spherharmonic(points, L)
    m = ComputeSphericalBasisAnalytical(points)
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

#Training loop


global t = 0.0
flux1 = zeros(ne, nx + 1, ny)
flux2 = zeros(ne, nx, ny + 1)

αT = zeros(nx*ny, ne)
phiT = zeros(nx*ny, ne)


    # Start Time loop

#begin
#for metaIter = 1:1
metaIter = 1 
#initial conditions
    
phi = zeros(Float32, ne, nx, ny)
phi_old = zeros(Float32, ne, nx, ny)
for j = 1:nx
    for i = 1:ny
        phi[1, i, j] = 1e-6
    end
end


anim = @animate for iter = 1:1000
    println("Iteration $(iter)")

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
    phi_old .= phi
    phi_train = Vector()
    h_train = Vector()
    errorsFound = false
    #####

    # realizability reconstruction
    error = 0
    @inbounds Threads.@threads for j = 1:ny
        for i = 1:nx
            phi[:, i, j] .= KitBase.realizable_reconstruct(α[:, i, j], m, weights, KitBase.maxwell_boltzmann_dual_prime)
            # print( phi[:, i, j])
            # print("|")
            # println( phi_old[:, i, j] )
            
            # Check the prediction error
            error = norm( phi[:, i, j]-phi_old[:, i, j],2)^2 # take care of the criterion
            if error > 0.0001
                #println(error)
                errorsFound = true
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
    end


    #retrain network on cells with too much error, if there is any
    if errorsFound
        print("Current number of  errorous datapoints: ")
        print(size(h_train)[1])
        print(" of ")
        print(nx*ny)
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
        model.fit(phiTrainMat,hTrainMat,epochs =50,batch_size = 32,validation_split=0.0,verbose= 0)
        println("model retrained")
        model.save("best_model.h5")
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
    contourf(pspace.x[1:nx, 1], pspace.y[1, 1:ny], phi[1, :, :])
end
cd(@__DIR__)
gif(anim, "lattice_mn_neural_reinforcement.gif")

#global t=0
#end
#println(phi[1, :, :])
#contourf(pspace.x[1:nx, 1], pspace.y[1, 1:ny], phi[1, :, :])


