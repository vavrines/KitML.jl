"""
KitML.jl : The lightweight module of neural differential equations within Kinetic.jl
Copyright (c) 2020 Tianbai Xiao <tianbaixiao@gmail.com>

"""

module KitML

using KitBase
using Flux
using DiffEqFlux

include("Neural/neural.jl")

end
