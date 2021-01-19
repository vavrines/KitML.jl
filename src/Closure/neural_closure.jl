using Flux
using Flux: @epochs



"""
    optimize_closure(α, m, ω, u, η::Function; optimizer=Newton())

Neural network for the entropy closure: `argmin(<η(α*m)> - α*u)`

"""
function call_neural_closure(X, model)
    """ 
    X: Model input
    model: neural network model
    """

    res = model(x)

    return res
end

function create_neural_closure(imputDim, outputDim)
    model = Chain(Dense(imputDim, 16, relu),Dense(16, 16,relu), Dense(16, outputDim))
    return model
end

function train_neural_closure(X, Y, model)
    """
    X: Model input data
    Y: Training data "truth"
    model: neural network model
    """

    function loss(x, y)
        sum(abs2, model(x) - y)
    end
    ps = Flux.params(model)
    data = [(X, Y)]
    cb() = @show(loss(X, Y))
    @epochs 10 Flux.train!(loss, ps, data, ADAM(), cb=cb)
    return model
end


