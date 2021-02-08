# ============================================================
# Neural Layers & Blocks
# ============================================================

export AbstractLayer, AbstractChain
export dense_layer
export FastAffine, Shortcut
export ICNNLayer, ICNNChain
export FastIC, FastICNN

abstract type AbstractLayer end
abstract type AbstractChain end

# ------------------------------------------------------------
# Extended Flux.Chain
# ------------------------------------------------------------
function Flux.Chain(D::Integer, N::Integer, σ::Function)
    t = ()
    for i = 1:N
        t = (t..., Dense(D, D, σ))
    end

    return Chain(t...)
end


"""
    dense_layer(
        in::T,
        out::T,
        σ = identity::Function;
        fw = randn::Function,
        fb = zeros::Function,
        isBias = true::Bool,
        precision = Float32,
    ) where {T<:Integer}

Create dense layer with meticulous settings

"""
function dense_layer(
    in::T,
    out::T,
    σ = identity::Function;
    fw = randn::Function,
    fb = zeros::Function,
    isBias = true::Bool,
    precision = Float32,
) where {T<:Integer}
    if isBias
        return Dense(fw(precision, out, in), fb(precision, out), σ)
    else
        return Dense(fw(precision, out, in), Flux.Zeros(precision, out), σ)
    end
end


"""
    struct FastAffine{I,F,F2} <: DiffEqFlux.FastLayer
        out::I
        in::I
        σ::F
        initial_params::F2
    end

Equivalent FastDense layer with controllable type

"""
struct FastAffine{I,F,F2} <: DiffEqFlux.FastLayer
    out::I
    in::I
    σ::F
    initial_params::F2
end

function FastAffine(
    in::Integer,
    out::Integer,
    σ = identity;
    fw = randn,
    fb = Flux.zeros,
    precision = Float32,
)
    initial_params() = vcat(vec(fw(precision, out, in)), fb(precision, out))
    return FastAffine(out, in, σ, initial_params)
end

(f::FastAffine)(x, p) =
    f.σ.(reshape(p[1:(f.out*f.in)], f.out, f.in) * x .+ p[(f.out*f.in+1):end])

DiffEqFlux.paramlength(f::FastAffine) = f.out * (f.in + 1)
DiffEqFlux.initial_params(f::FastAffine) = f.initial_params()


"""
    struct Shortcut{T}
        chain::T
        f::Function
        σ::Function
    end

Shortcut connection for ResNet-type blocks

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
    z = σ.(softplus.(W) * z + U * x .+ b)

    return reshape(z, :, sz[2:end]...)
end

Flux.@functor ICNNLayer

function Base.show(io::IO, model::ICNNLayer{T1,T2,T3}) where {T1,T2,T3}
    print(
        io,
        "ICNN layer{$T1,$T2,$T3}\n",
        "nonnegative weights for: $(model.W |> size)\n",
        "input weights: $(model.U |> size)\n",
        "bias: $(model.b |> size)\n",
        "activation: $(model.σ)\n",
    )
end


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
            HLayers = (
                HLayers...,
                KitML.ICNNLayer(
                    layer_sizes[i],
                    din,
                    out,
                    activation;
                    fw = fw,
                    fb = fb,
                    precision = precision,
                ),
            )
            i += 1
        end
    end
    HLayers = (
        HLayers...,
        KitML.ICNNLayer(
            layer_sizes[end],
            din,
            dout,
            identity;
            fw = fw,
            fb = fb,
            precision = precision,
        ),
    )


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

function Base.show(io::IO, model::ICNNChain{T1,T2}) where {T1,T2}
    print(
        io,
        "ICNN chain{$T1,$T2}\n",
        "input layer: $(model.InLayer)\n",
        "hidden layer: $(model.HLayer |> length) ICNN layers\n",
    )
end

struct FastIC{I<:Integer,F1,F2} <: DiffEqFlux.FastLayer
    zin::I
    xin::I
    out::I
    σ::F1
    initial_params::F2
end

function FastIC(
    zin::Integer,
    xin::Integer,
    out::Integer,
    σ = identity;
    fw = randn,
    fb = zeros,
    precision = Float32,
)
    initial_params() =
        vcat(vec(fw(precision, out, zin)), vec(fw(precision, out, xin)), fb(precision, out))
    return FastIC{typeof(out),typeof(σ),typeof(initial_params)}(
        zin,
        xin,
        out,
        σ,
        initial_params,
    )
end

function (f::FastIC)(z, x, p)
    f.σ.(
        softplus.(reshape(p[1:(f.out*f.zin)], f.out, f.zin)) * z .+
        reshape(p[f.out*f.zin+1:(f.out*f.zin+f.out*f.xin)], f.out, f.xin) * x .+
        p[(f.out*f.zin+f.out*f.xin+1):end],
    )
end

DiffEqFlux.paramlength(f::FastIC) = f.out * (f.zin + f.xin + 1)
DiffEqFlux.initial_params(f::FastIC) = f.initial_params()

struct FastICNN{T} <: DiffEqFlux.FastLayer
    layers::T
end

function FastICNN(
    din::TI,
    dout::TI,
    layer_sizes::TT,
    activation = identity::Function;
    fw = randn::Function,
    fb = zeros::Function,
    precision = Float32,
) where {TI<:Integer,TT<:Union{Tuple,AbstractVector}}

    layers = (FastDense(din, layer_sizes[1], activation),)

    if length(layer_sizes) > 1
        i = 1
        for out in layer_sizes[2:end]
            layers = (
                layers...,
                FastIC(
                    layer_sizes[i],
                    din,
                    out,
                    activation;
                    fw = fw,
                    fb = fb,
                    precision = precision,
                ),
            )
            i += 1
        end
    end
    layers = (
        layers...,
        FastIC(
            layer_sizes[end],
            din,
            dout,
            identity;
            fw = fw,
            fb = fb,
            precision = precision,
        ),
    )

    return FastICNN(layers)

end

DiffEqFlux.initial_params(c::FastICNN) = vcat(initial_params.(c.layers)...)

function (m::FastICNN)(x::AbstractArray, p)
    z = m.layers[1](x, p[1:DiffEqFlux.paramlength(m.layers[1])])
    counter = DiffEqFlux.paramlength(m.layers[1])
    for i = 2:length(m.layers)
        z = m.layers[i](z, x, p[counter+1:counter+DiffEqFlux.paramlength(m.layers[i])])
        counter += DiffEqFlux.paramlength(m.layers[i])
    end

    return z
end
