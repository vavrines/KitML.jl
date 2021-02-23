# ============================================================
# Neural & Universal Closures
# ============================================================

export neural_closure, create_neural_closure

"""
    neural_closure(X, model)

Neural closure model computation

@args X: model input
@args model: neural network model
"""
function neural_closure(model, X)
    return model(X)
end

function neural_closure(model::PyObject, X)
    py"""
    import tensorflow as tf

    def dnn(model, input):
        '''
        Input: input.shape = (nCells,nMaxMoment), nMaxMoment = 4 in case of MK7
        Output: Gradient of the network wrt input
        '''
        # predictions = model.predict(input)

        x_model = tf.Variable(input)

        with tf.GradientTape() as tape:
            #predictions = model.predict(x_model)
            predictions = model(x_model)

        gradients = tape.gradient(predictions, x_model)

        return gradients.numpy()
    """

    return py"dnn"(model, X)
end


"""
    create_neural_closure(imputDim, outputDim; acfun = relu)

Create neural closure model
"""
function create_neural_closure(
    Din::T,
    Dout::T,
    Dhid = 10::T,
    Nhid = 1::T;
    acfun = relu,
    mode = :icnn,
) where {T<:Integer}
    if mode == :dense
        # standard model
        model = Chain(Dense(Din, Dhid, acfun), Chain(Dhid, Nhid, acfun), Dense(Dhid, Dout))
    elseif mode == :icnn
        # ICNN model
        model = ICNNChain(Din, Dout, ones(Int, Nhid) * Dhid, acfun)
    end

    return model
end
