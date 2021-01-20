# ============================================================
# Training and Optimization Methods
# ============================================================

export sci_train!

"""
    sci_train!(ann, data, opt = ADAM(), ne = 1, nb = 256)
    sci_train!(ann, dl::Flux.Data.DataLoader, opt = ADAM(), ne = 1)

Scientific machine learning training function

@args ann: neural network model
@args data: tuple (X, Y) of dataset
@args opt: optimizer 
@args ne: epoch number
@args nb: batch size
"""
function sci_train!(ann, data, opt = ADAM(), ne = 1, nb = 256)
    L = size(data[1], 2)
    loss(x, y) = sum(abs2, ann(x) - y) / L
    ps = params(ann)
    cb = () -> println("loss: $(loss(data[1], data[2]))")

    Flux.@epochs ne Flux.train!(
        loss,
        ps,
        Flux.Data.DataLoader(data[1], data[2], batchsize = nb, shuffle = true),
        opt,
        cb = Flux.throttle(cb, 1),
    )

    return nothing
end

function sci_train!(ann, dl::Flux.Data.DataLoader, opt = ADAM(), ne = 1)
    L = size(dl.data[1], 2)
    loss(x, y) = sum(abs2, ann(x) - y) / L
    ps = params(ann)
    cb = () -> println("loss: $(loss(dl.data[1], dl.data[2]))")

    Flux.@epochs ne Flux.train!(
        loss,
        ps,
        dl,
        opt,
        cb = Flux.throttle(cb, 1),
    )

    return nothing
end
