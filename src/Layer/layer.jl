# ============================================================
# Neural Layers & Blocks
# ============================================================

export Shortcut

"""
Shortcut connection for ResNet-type blocks

@vars chain: inner chain of layer(s)
@vars f: connection function between chain and shortcut inputs
@vars σ: activation function
"""
struct Shortcut{T}
    chain::T
    f::Function
    σ::Function
end

Shortcut(chain::T) where {T} = Shortcut{typeof(chain)}(chain, +, tanh)

Flux.@functor Shortcut

(nn::Shortcut)(x) = nn.σ.(nn.f(nn.chain(x), x))

function Base.show(io::IO, model::Shortcut{T}) where {T}
    print(
        io,
        "Shortcut{$T}\n",
        "chain: $(model.chain)\n",
        "connection: $(model.f)\n",
        "activation: $(model.σ)\n",
    )
end


"""
Shortcut connection for ICNN approach by Amos et al.

@vars chain: inner chain of layer(s)
@vars f: connection function between chain and shortcut inputs
@vars σ: activation function
"""

"""
    struct ICNNLayer{T1<:AbstractArray,T2<:Union{Flux.Zeros, AbstractVector},T3}
        W::T1
        U::T1
        b::T2
        σ::T3
    end

Input Convex Layer

"""
struct ICNNLayer{T1<:AbstractArray,T2<:Union{Flux.Zeros, AbstractVector},T3}
    W::T1
    U::T1
    b::T2
    σ::T3
end

# constructor
ICNNLayer(z_in::Integer, x_in::Integer, out::Integer, activation = identity::Function) =
    ICNNLayer(randn(out, z_in), randn(out, x_in), randn(out), activation)

# forward pass
(m::ICNNLayer)(x) = m.σ.(m.W * x + m.b)
(m::ICNNLayer)(z, x) = m.σ.(m.W * z + softplus.(m.U) * x + m.b)

# track params
Flux.@functor ICNNLayer


"""
Input Convex Neural Network (ICNN)

"""
struct ICNN{T1,T2,T3}
    InLayer::T1
    HLayer1::T2
    HLayer2::T2
    σ::T3
end

# constructor
ICNN(input_dim::Integer, output_Dim::Integer, layer_sizes::Vector, activation = tanh::Function) = begin
    InLayer = Dense(input_dim, layer_sizes[1])
    HLayers = []
    if length(layer_sizes) > 1
        i = 1
        for out in layer_sizes[2:end]
            push!(HLayers, ICNNLayer(layer_sizes[i], input_dim, out, activation))
            i += 1
        end
        push!(HLayers, ICNNLayer(layer_sizes[end], input_dim, output_Dim, activation))
    end
    ICNN(InLayer, HLayers[1], HLayers[2], activation)
end

# forward pass
(m::ICNN)(x) = begin
    z = m.σ.(m.InLayer(x))
    z = m.HLayer1(z, x)
    z = m.HLayer2(z, x)
    return z
end

Flux.@functor ICNN


"""
Revised Input Convex Neural Network (ICNN)

"""
struct ICNNChain{T1,T2}
    InLayer::T1
    HLayer::T2
end

function ICNNChain(
    din::TI,
    dout::TI,
    layer_sizes::TT,
    activation = identity::Function,
) where {TI<:Integer,TT<:Union{Tuple,AbstractVector}}
    
    InLayer = Dense(din, layer_sizes[1], activation)

    HLayers = ()
    if length(layer_sizes) > 1
        i = 1
        for out in layer_sizes[2:end]
            HLayers = (HLayers..., KitML.ICNNLayer(layer_sizes[i], din, out, activation))
            i += 1
        end
        HLayers = (HLayers..., KitML.ICNNLayer(layer_sizes[end], din, dout, activation)) # @FIXME activation or identity here?
    end

    return ICNNChain(InLayer, HLayers)

end

(m::ICNNChain)(x) = begin
    z = m.InLayer(x)
    for i in eachindex(m.HLayer)
        z = m.HLayer[i](z, x)
    end
    return z
end

Flux.@functor ICNNChain
