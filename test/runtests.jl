using Test, KitML
using KitBase, Solaris
using Solaris.Flux

cd(@__DIR__)

include("test_data.jl")
include("test_equation.jl")
include("test_solver.jl")
include("test_closure.jl")
include("test_layer.jl")
