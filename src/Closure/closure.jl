# ============================================================
# Neural & Universal Closures
# ============================================================

export neural_closure, create_neural_closure

"""
$(SIGNATURES)

Neural closure model computation

# Arguments
* ``model``: neural network model
* ``X``: model input
"""
function neural_closure(model, X)
    return Zygote.gradient(x -> first(model(x)), X)[1]
end

"""
$(SIGNATURES)
"""
function neural_closure(model, p, X)
    return Zygote.gradient(x -> first(model(x, p)), X)[1]
end

"""
$(SIGNATURES)
"""
function neural_closure(model::PyObject, X)
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
$(SIGNATURES)

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
