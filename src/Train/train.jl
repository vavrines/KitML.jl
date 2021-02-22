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
- device: cpu / gpu
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
    device = cpu,
    maxiters = 200::Integer,
    kwargs...,
) where {T<:DiffEqFlux.FastLayer}
    data = data |> device
    θ = θ |> device
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
    sci_train!(ann, data::Tuple, opt = ADAM(); device = cpu, epoch = 1, batch = 1)
    sci_train!(ann, dl::Flux.Data.DataLoader, opt = ADAM(); device = cpu, epoch = 1)

Scientific machine learning training function

- @args ann: neural network model
- @args data: tuple (X, Y) of dataset
- @args opt: optimizer 
- @args epoch: epoch number
- @args batch: batch size
- @args device: cpu / gpu
"""
function sci_train!(ann, data::Tuple, opt = ADAM(); device = cpu, epoch = 1, batch = 1)
    X, Y = data |> device
    L = size(X, 2)
    data = Flux.Data.DataLoader(X, Y, batchsize = batch, shuffle = true) |> device

    ann = device(ann)
    ps = params(ann)
    loss(x, y) = sum(abs2, ann(x) - y) / L
    cb = () -> println("loss: $(loss(X, Y))")

    Flux.@epochs epoch Flux.train!(loss, ps, data, opt, cb = Flux.throttle(cb, 1))

    return nothing
end

function sci_train!(ann, dl::Flux.Data.DataLoader, opt = ADAM(); device = cpu, epoch = 1)
    X, Y = dl.data |> device
    L = size(X, 2)
    dl = dl |> device

    ann = device(ann)
    ps = params(ann)
    loss(x, y) = sum(abs2, ann(x) - y) / L
    cb = () -> println("loss: $(loss(X, Y))")

    Flux.@epochs epoch Flux.train!(loss, ps, dl, opt, cb = Flux.throttle(cb, 1))

    return nothing
end

# ------------------------------------------------------------
# TensorFlow
# ------------------------------------------------------------
function sci_train!(ann::PyObject, data::Tuple; device = cpu, epoch = 1)
    X, Y = data
    ann.fit(X, Y, epochs = epoch)

    return nothing
end
