# KitML.jl

[![version](https://juliahub.com/docs/KitML/version.svg)](https://juliahub.com/ui/Packages/KitML/akJVY)
![CI](https://github.com/vavrines/KitML.jl/workflows/CI/badge.svg)
[![](https://img.shields.io/badge/docs-stable-green.svg)](https://xiaotianbai.com/Kinetic.jl/stable/)
[![](https://img.shields.io/badge/docs-dev-green.svg)](https://xiaotianbai.com/Kinetic.jl/dev/)
[![codecov](https://codecov.io/gh/vavrines/KitML.jl/branch/main/graph/badge.svg?token=OnazyqLA4K)](https://codecov.io/gh/vavrines/KitML.jl)
[![deps](https://juliahub.com/docs/KitML/deps.svg)](https://juliahub.com/ui/Packages/KitML/akJVY?t=2)
[![GitHub commits since tagged version](https://img.shields.io/github/commits-since/vavrines/KitML.jl/v0.3.0.svg?style=social&logo=github)](https://github.com/vavrines/KitML.jl)

**KitML.jl** serves as a lightweight module of neural differential equations in [Kinetic.jl](https://github.com/vavrines/Kinetic.jl) ecosystem.
The package is interested in theoretical and numerical studies of many-particle systems of gases, photons, plasmas, neutrons, etc.
It employs the finite volume method (FVM) to conduct 1-3 dimensional numerical simulations on CPUs and GPUs.
Any advection-diffusion-type equation can be solved within the framework.
Special attentions have been paid on Hilbert's sixth problem, i.e. to build the numerical passage between kinetic theory of gases, e.g. the Boltzmann equation

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;f}{\partial&space;t}&plus;\mathbf{v}&space;\cdot&space;\nabla_{\mathbf{x}}&space;f&space;=&space;\int_{\mathbb&space;R^3}&space;\int_{\mathcal&space;S^2}&space;\mathcal&space;B(\cos&space;\beta,&space;|\mathbf{v}-\mathbf{v_*}|)&space;\left[&space;f(\mathbf&space;v')f(\mathbf&space;v_*')-f(\mathbf&space;v)f(\mathbf&space;v_*)\right]&space;d\mathbf&space;\Omega&space;d\mathbf&space;v_*" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\frac{\partial&space;f}{\partial&space;t}&plus;\mathbf{v}&space;\cdot&space;\nabla_{\mathbf{x}}&space;f&space;=&space;\int_{\mathbb&space;R^3}&space;\int_{\mathcal&space;S^2}&space;\mathcal&space;B(\cos&space;\beta,&space;|\mathbf{v}-\mathbf{v_*}|)&space;\left[&space;f(\mathbf&space;v')f(\mathbf&space;v_*')-f(\mathbf&space;v)f(\mathbf&space;v_*)\right]&space;d\mathbf&space;\Omega&space;d\mathbf&space;v_*" title="\frac{\partial f}{\partial t}+\mathbf{v} \cdot \nabla_{\mathbf{x}} f = \int_{\mathbb R^3} \int_{\mathcal S^2} \mathcal B(\cos \beta, |\mathbf{v}-\mathbf{v_*}|) \left[ f(\mathbf v')f(\mathbf v_*')-f(\mathbf v)f(\mathbf v_*)\right] d\mathbf \Omega d\mathbf v_*" /></a>

and continuum mechanics, e.g. the Euler and Navier-Stokes equations

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;\mathbf&space;W}{\partial&space;t}&space;&plus;&space;\nabla_\mathbf&space;x&space;\cdot&space;\mathbf&space;F&space;=&space;\mathbf&space;S" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\frac{\partial&space;\mathbf&space;W}{\partial&space;t}&space;&plus;&space;\nabla_\mathbf&space;x&space;\cdot&space;\mathbf&space;F&space;=&space;\mathbf&space;S" title="\frac{\partial \mathbf W}{\partial t} + \nabla_\mathbf x \cdot \mathbf F = \mathbf S" /></a>

A partial list of current supported neural and universal models include
- neural network enhanced Boltzmann equation
- neural entropy closure hierarchies

## Documentation

For the detailed information on the implementation and usage of the package, please
[check the documentation](https://xiaotianbai.com/Kinetic.jl/dev/).

## Contributing

If you have further questions regarding KitML.jl or have got an idea on improving it, please feel free to get in touch. Open an issue or pull request if you'd like to work on a new feature or even if you're new to open-source and want to find a cool little project or issue to work on that fits your interests. We're more than happy to help along the way.