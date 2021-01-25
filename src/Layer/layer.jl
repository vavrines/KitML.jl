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
