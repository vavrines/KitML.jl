"""
KitML.jl : The lightweight module of neural differential equations in Kinetic.jl

Copyright (c) 2021 Tianbai Xiao <tianbaixiao@gmail.com>
"""

module KitML

using KitBase
using Flux
using DiffEqFlux
using OrdinaryDiffEq

include("Equation/equation.jl")
include("Neural/neural.jl")

end
