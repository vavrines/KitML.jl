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
function neural_closure(model, X::T) where {T<:AbstractArray}
    return Zygote.gradient(x -> first(model(x)), X)[1]
end

function neural_closure(model::T, p, X::T1) where {T<:DiffEqFlux.FastLayer,T1<:AbstractArray}
    return Zygote.gradient(x -> first(model(x, p)), X)[1]
end

function neural_closure(model::PyObject, X::T) where {T<:AbstractArray}
    py"""
    import tensorflow as tf

    def dnn(model, input):
        '''
        Input: input.shape = (nCells,nMaxMoment), nMaxMoment = 4 in case of MK7
        Output: Gradient of the network wrt input
        '''
        x_model = tf.Variable(input)
        with tf.GradientTape() as tape:
            predictions = model(x_model) # model.predict(x_model) doesn't work
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
    return ICNN(Din, Dout, ones(Int, Nhid) * Dhid, acfun)
end
