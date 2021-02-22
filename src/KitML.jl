"""
KitML.jl : The lightweight module of neural differential equations in Kinetic.jl

Copyright (c) 2021 Tianbai Xiao & Steffen Schotthöfer <tianbaixiao@gmail.com>
"""

module KitML

using KitBase
using KitBase.JLD2
using KitBase.PyCall
using Flux
using DiffEqFlux
using GalacticOptim
using OrdinaryDiffEq
using CSV
using DataFrames

include("IO/io.jl")
include("Equation/equation.jl")
include("Layer/layer.jl")
include("Train/train.jl")
include("Closure/closure.jl")
include("Solver/solver.jl")

end
