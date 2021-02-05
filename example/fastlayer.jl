using DiffEqFlux, Flux

struct FastAffine{I,F,F2} <: DiffEqFlux.FastLayer
    out::I
    in::I
    σ::F
    initial_params::F2

    function FastAffine(in::Integer, out::Integer, σ = identity;
                   initW = Flux.glorot_uniform, initb = Flux.zeros)
        initial_params() = vcat(vec(initW(out, in)), initb(out))
        new{typeof(out),typeof(σ),typeof(initial_params)}(out, in, σ, initial_params)
    end
end
(f::FastAffine)(x,p) = f.σ.(reshape(p[1:(f.out*f.in)],f.out,f.in)*x .+ p[(f.out*f.in+1):end])

DiffEqFlux.paramlength(f::FastAffine) = f.out*(f.in + 1)
DiffEqFlux.initial_params(f::FastAffine) = f.initial_params()

nn = FastChain(FastAffine(4, 4, tanh), FastAffine(4, 4))
p = initial_params(nn)

X = randn(4, 10)
Y = rand(4, 10)

nn(X, p)

function loss(p)
    loss = sum(abs2, nn(X, p) .- Y)
end

cb = function (p, l)
    println("loss: $l")
    return false
end

res = DiffEqFlux.sciml_train(loss, p, ADAM(), cb = cb, maxiters = 100)
res = DiffEqFlux.sciml_train(loss, res.minimizer, ADAM(), cb = cb, maxiters = 100)