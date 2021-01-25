# ============================================================
# Neural & Universal Closures
# ============================================================

export neural_closure,
       create_neural_closure,
       train_neural_closure

"""
    neural_closure(X, model)

Neural closure model computation

@args X: model input
@args model: neural network model
"""
function neural_closure(X, model)
    res = model(X)

    return res
end


"""
    create_neural_closure(imputDim, outputDim; acfun = relu)

Create neural closure model
"""
function create_neural_closure(imputDim, outputDim; acfun = relu)

    # standard model
    #model = Chain(Dense(imputDim, 16, acfun), Dense(16, 16, acfun), Dense(16, outputDim))

    # ICNN model
    model = ICNN(imputDim,outputDim,[10,10,10],acfun)

    return model
end


"""
    train_neural_closure(X, Y, model)

Continuous training based on existing model

@args X: Model input data
@args Y: Training data "truth"
@args model: neural network model
@args ne: number of epochs
"""
function train_neural_closure(X, Y, model, ne = 10)
    loss(x, y) = sum(abs2, model(x) - y)
    ps = Flux.params(model)
    data = [(X, Y)]
    Flux.@epochs ne Flux.train!(loss, ps, data, ADAM())

    return model
end


"""
function ResNet_closure(imputDim, outputDim)
end
"""
