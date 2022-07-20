"""
KitML.jl: The lightweight module of neural differential equations in Kinetic.jl

Copyright (c) 2021-2022 Tianbai Xiao & Steffen Schotthöfer <tianbaixiao@gmail.com>
"""

module KitML

using LinearAlgebra
using KitBase
using KitBase.JLD2
using KitBase.PyCall
using KitBase.Reexport
@reexport using Solaris
using Solaris.DiffEqFlux

include("Data/data.jl")
include("Equation/equation.jl")
include("Closure/closure.jl")
include("Solver/solver.jl")

end
