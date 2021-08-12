"""
KitML.jl: The lightweight module of neural differential equations in Kinetic.jl

Copyright (c) 2021 Tianbai Xiao & Steffen Schotthöfer <tianbaixiao@gmail.com>
"""

module KitML

using KitBase
using KitBase.JLD2
using KitBase.PyCall
using KitBase.Reexport
@reexport using Solaris
using Solaris.DiffEqFlux

include("Equation/equation.jl")
include("Closure/closure.jl")
include("Solver/solver.jl")

end
