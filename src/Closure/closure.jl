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
function neural_closure(X, model)
    res = model(X)

    return res
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
