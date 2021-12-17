using Test, KitML
using Solaris
using Solaris.Flux, Solaris.DiffEqFlux

cd(@__DIR__)

include("test_data.jl")
include("test_equation.jl")
include("test_solver.jl")
include("test_closure.jl")
