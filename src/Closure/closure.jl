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

################################################################################
# input convex neural network
struct ICNNLayer
    W
    U
    b
    act
end

# constructor
ICNNLayer(z_in::Integer, x_in::Integer, out::Integer, activation) =
    ICNNLayer(randn(out, z_in), randn(out, x_in), randn(out), activation)
# forward pass
(m::ICNNLayer)(z, x) = m.act(m.W*z + softplus.(m.U)*x + m.b)
# track params
Flux.@functor ICNNLayer

# Input Convex Neural Network (ICNN)
struct ICNN
    InLayer
    HLayer1
    HLayer2
    act
end
# constructor
ICNN(input_dim::Integer, layer_sizes::Vector, activation) = begin
    InLayer = Dense(input_dim, layer_sizes[1])
    HLayers = []
    if length(layer_sizes) > 1
        i = 1
        for out in layer_sizes[2:end]
            push!(HLayers, ICNNLayer(layer_sizes[i], input_dim, out, activation))
            i += 1
        end
        push!(HLayers, ICNNLayer(layer_sizes[end], input_dim, 1, activation))
    end
    ICNN(InLayer, HLayers[1], HLayers[2], activation)
end
# forward pass
(m::ICNN)(x) = begin
    z = m.act(m.InLayer(x))
    z = m.HLayer1(z, x)
    z = m.HLayer2(z, x)
    return z
end
Flux.@functor ICNN

function create_neural_closure(imputDim, outputDim; acfun = relu)

    # standard model
    # model = Chain(Dense(imputDim, 16, acfun), Dense(16, 16, acfun), Dense(16, outputDim))

    # ICNN model
    model = ICNN(imputDim,4,softplus)

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
