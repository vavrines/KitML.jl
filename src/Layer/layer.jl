# ============================================================
# Neural Layers & Blocks
# ============================================================

export AbstractLayer, AbstractChain
export Shortcut, ICNNLayer, ICNNChain

abstract type AbstractLayer end
abstract type AbstractChain end


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
    struct ICNNLayer{T1<:AbstractArray,T2<:Union{Flux.Zeros, AbstractVector},T3}
        W::T1
        U::T1
        b::T2
        σ::T3
    end

Input Convex Neural Network (ICNN) Layer by Amos et al.

"""
struct ICNNLayer{T1<:AbstractArray,T2<:Union{Flux.Zeros,AbstractVector},T3} <: AbstractLayer
    W::T1
    U::T1
    b::T2
    σ::T3
end

function ICNNLayer(
    z_in::T,
    x_in::T,
    out::T,
    activation = identity::Function;
    fw = randn::Function,
    fb = zeros::Function,
    precision = Float32,
) where {T<:Integer}
    return ICNNLayer(
        fw(precision, out, z_in),
        fw(precision, out, x_in),
        fb(precision, out),
        activation,
    )
end

function (m::ICNNLayer)(x::AbstractArray)
    W, b, σ = m.W, m.b, m.σ
    sz = size(x)
    x = reshape(x, sz[1], :) # reshape to handle dims > 1 as batch dimensions 
    x = σ.(W * x .+ b)

    return reshape(x, :, sz[2:end]...)   
end

function (m::ICNNLayer)(z::AbstractArray, x::AbstractArray)
    W, U, b, σ = m.W, m.U, m.b, m.σ
    sz = size(z)
    sx = size(x)
    z = reshape(z, sz[1], :)
    x = reshape(x, sx[1], :)
    z = σ.(W * z + softplus.(U) * x .+ b)

    return reshape(z, :, sz[2:end]...)   
end

Flux.@functor ICNNLayer

#=
"""
Input Convex Neural Network (ICNN) Layer by Amos et al.
**deprecated**
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
=#

"""
    struct ICNNChain{T1,T2}
        InLayer::T1
        HLayer::T2
    end

Input Convex Neural Network (ICNN) Layer by Amos et al.

"""
struct ICNNChain{T1,T2} <: AbstractChain
    InLayer::T1
    HLayer::T2
end

function ICNNChain(
    din::TI,
    dout::TI,
    layer_sizes::TT,
    activation = identity::Function;
    fw = randn::Function,
    fb = zeros::Function,
    precision = Float32,
) where {TI<:Integer,TT<:Union{Tuple,AbstractVector}}
    
    InLayer = Dense(din, layer_sizes[1], activation)

    HLayers = ()
    if length(layer_sizes) > 1
        i = 1
        for out in layer_sizes[2:end]
            HLayers = (HLayers..., KitML.ICNNLayer(layer_sizes[i], din, out, activation; fw = fw, fb = fb, precision = precision))
            i += 1
        end
        HLayers = (HLayers..., KitML.ICNNLayer(layer_sizes[end], din, dout, identity; fw = fw, fb = fb, precision = precision))
    end

    return ICNNChain(InLayer, HLayers)

end

(m::ICNNChain)(x::AbstractArray) = begin
    z = m.InLayer(x)
    for i in eachindex(m.HLayer)
        z = m.HLayer[i](z, x)
    end
    return z
end

Flux.@functor ICNNChain
