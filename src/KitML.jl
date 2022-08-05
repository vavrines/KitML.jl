"""
KitML.jl: The lightweight module of neural differential equations in Kinetic.jl

Copyright (c) 2021-2022 Tianbai Xiao & Steffen Schotthöfer <tianbaixiao@gmail.com>
"""

module KitML

using LinearAlgebra
using KitBase
using KitBase.FiniteMesh.DocStringExtensions
using KitBase.JLD2
using KitBase.PyCall
using KitBase.Reexport
using KitBase: AV, AM, AVOM
@reexport using Solaris
using Solaris.Flux
using Solaris.Zygote

include("Data/data.jl")
include("Equation/equation.jl")
include("Closure/closure.jl")
include("Layer/layer.jl")
include("Solver/solver.jl")

end
