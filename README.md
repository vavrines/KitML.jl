# KitML.jl

[![version](https://juliahub.com/docs/KitML/version.svg)](https://juliahub.com/ui/Packages/KitML/akJVY)
![CI](https://img.shields.io/github/workflow/status/vavrines/KitML.jl/CI)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://xiaotianbai.com/Kinetic.jl/stable/)
[![codecov](https://img.shields.io/codecov/c/github/vavrines/KitML.jl)](https://codecov.io/gh/vavrines/KitML.jl)
[![deps](https://juliahub.com/docs/KitML/deps.svg)](https://juliahub.com/ui/Packages/KitML/akJVY?t=2)
[![GitHub commits since tagged version](https://img.shields.io/github/commits-since/vavrines/KitML.jl/v0.4.0.svg?style=social&logo=github)](https://github.com/vavrines/KitML.jl)

This lightweight module provides neural differential equations for Kinetic.jl ecosystem.
The finite volume method (FVM) is employed to perform 1-3 dimensional numerical simulations on CPUs and GPUs.
Machine learning methods are seamlessly integrated to build data-drivenclosure models and accelerate the computation of nonlinear terms.
A partial list of current supported models and equations include:
- Boltzmann equation
- radiative transfer equation
- Fokker-Planck-Landau equation
- direct simulation Monte Carlo
- advection-diffusion equation
- Burgers equation
- Euler equations
- Navier-Stokes equations
- Magnetohydrodynamical equations
- Maxwell's equations

For the detailed information on the implementation and usage of the package, you may
[check the documentation](https://xiaotianbai.com/Kinetic.jl/dev/).