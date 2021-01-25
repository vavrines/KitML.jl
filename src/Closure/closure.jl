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
function create_neural_closure(Din, Dout, Dhid = 10; acfun = relu, mode = :icnn)
    if mode == :dense
        # standard model
        model = Chain(Dense(Din, Dhid, acfun), Dense(Dhid, Dhid, acfun), Dense(Dhid, Dout))
    elseif mode == :icnn
        # ICNN model
        model = ICNNChain(Din, Dout, [Dhid, Dhid, Dhid], acfun)
    end

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
    sci_train!(model, (X, Y))
    return model
end


"""
function ResNet_closure(imputDim, outputDim)
end
"""
