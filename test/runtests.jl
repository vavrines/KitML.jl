using Test, KitML
using Solaris
using Solaris.Flux, Solaris.DiffEqFlux

const SR = Solaris

include("test_data.jl")
include("test_io.jl")
include("test_train.jl")
include("test_equation.jl")
include("test_solver.jl")
include("test_closure.jl")
