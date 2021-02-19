# ============================================================
# Training & Optimization Methods
# ============================================================

export sci_train, sci_train!

"""
    sci_train(
        ann::T,
        data,
        θ = initial_params(ann),
        opt = ADAM(),
        adtype = GalacticOptim.AutoZygote(),
        args...;
        maxiters = 200::Integer,
        kwargs...,
    ) where {T<:DiffEqFlux.FastLayer}

Scientific machine learning training function

- ann: neural network model
- data: tuple (X, Y) of dataset
- θ: parameters of neural network
- opt: optimizer
- adtype: automatical differentiation type
- args: rest arguments
- maxiters: maximal iteration number
- kwargs: keyword arguments
"""
function sci_train(
    ann::T,
    data,
    θ = initial_params(ann),
    opt = ADAM(),
    adtype = GalacticOptim.AutoZygote(),
    args...;
    maxiters = 200::Integer,
    kwargs...,
) where {T<:DiffEqFlux.FastLayer}
    L = size(data[1], 2)
    loss(p) = sum(abs2, ann(data[1], p) - data[2]) / L

    cb = function (p, l)
        println("loss: $(loss(p))")
        return false
    end

    f = GalacticOptim.OptimizationFunction((x, p) -> loss(x), adtype)
    fi = GalacticOptim.instantiate_function(f, θ, adtype, nothing)
    prob = GalacticOptim.OptimizationProblem(fi, θ; kwargs...)

    return GalacticOptim.solve(prob, opt, args...; cb = Flux.throttle(cb, 1), maxiters = maxiters, kwargs...)
end


"""
    sci_train!(ann, data, opt = ADAM(), ne = 1, nb = 256; device = cpu)
    sci_train!(ann, dl::Flux.Data.DataLoader, opt = ADAM(), ne = 1; device = cpu)

Scientific machine learning training function

- @args ann: neural network model
- @args data: tuple (X, Y) of dataset
- @args opt: optimizer 
- @args ne: epoch number
- @args nb: batch size
- @args device: cpu / gpu
"""
function sci_train!(ann, data::Tuple, opt = ADAM(), ne = 1, nb = 256; device = cpu)
    X, Y = data |> device
    L = size(X, 2)
    data = Flux.Data.DataLoader(X, Y, batchsize = nb, shuffle = true) |> device

    ann = device(ann)
    ps = params(ann)
    loss(x, y) = sum(abs2, ann(x) - y) / L
    cb = () -> println("loss: $(loss(X, Y))")

    Flux.@epochs ne Flux.train!(loss, ps, data, opt, cb = Flux.throttle(cb, 1))

    return nothing
end

function sci_train!(ann, dl::Flux.Data.DataLoader, opt = ADAM(), ne = 1; device = cpu)
    X, Y = dl.data |> device
    L = size(X, 2)
    dl = dl |> device

    ann = device(ann)
    ps = params(ann)
    loss(x, y) = sum(abs2, ann(x) - y) / L
    cb = () -> println("loss: $(loss(X, Y))")

    Flux.@epochs ne Flux.train!(loss, ps, dl, opt, cb = Flux.throttle(cb, 1))

    return nothing
end
