# ============================================================
# Neural & Universal Differential Equations
# ============================================================

export ube_dfdt, ube_dfdt!

"""
$(SIGNATURES)

Right-hand side of universal Boltzmann equation

# Arguments
* ``f``: particle distribution function in 1D formulation
* ``p``: M, τ, ann (network & parameters)
* ``t``: time span
"""
function ube_dfdt(f, p, t)
    M, τ, ann = p

    if ann[1] isa FnChain
        df = (M - f) / τ + ann[1](M - f, ann[2])
    elseif ann[1] isa Chain
        df = (M - f) / τ + ann[1](M - f)
    end

    return df
end


"""
$(SIGNATURES)

Right-hand side of universal Boltzmann equation

# Arguments
* ``df``: derivatives of particle distribution function
* ``f``: particle distribution function in 1D formulation
* ``p``: M, τ, ann (network & parameters)
* ``t``: time span
"""
function ube_dfdt!(df, f, p, t)
    M, τ, ann = p

    if ann[1] isa FnChain
        df .= (M - f) / τ + ann[1](M - f, ann[2])
    elseif ann[1] isa Chain
        df .= (M - f) / τ + ann[1](M - f)
    end
end
