"""
KitML.jl : The lightweight module of neural differential equations in Kinetic.jl

Copyright (c) 2021 Tianbai Xiao <tianbaixiao@gmail.com>
"""

module KitML

using KitBase
using Flux
using DiffEqFlux
using OrdinaryDiffEq
using JLD2

include("IO/io.jl")
include("Layer/layer.jl")
include("Equation/equation.jl")
include("Solver/solver.jl")
include("Closure/closure.jl")

end
