vs = KitML.KitBase.VSpace1D(-5, 5, 16; precision = Float32)

#--- class 1 ---#
# full distribution
mn = FnChain(FnDense(vs.nu, vs.nu, tanh; bias = false), FnDense(vs.nu, vs.nu; bias = false))
νn = FnChain(
    FnDense(vs.nu + 1, vs.nu, tanh; bias = false),
    FnDense(vs.nu, vs.nu, relu; bias = false),
)
nn = BGKNet(mn, νn)
p = init_params(nn)

nn(rand(Float32, vs.nu + 1, 2), p, vs, 3)

# reduced distribution
mn = FnChain(
    FnDense(vs.nu * 2, vs.nu * 2, tanh; bias = false),
    FnDense(vs.nu * 2, vs.nu * 2; bias = false),
)
νn = FnChain(
    FnDense(vs.nu * 2 + 1, vs.nu * 2 + 1, tanh; bias = false),
    FnDense(vs.nu * 2 + 1, vs.nu * 2, relu; bias = false),
)
nn = BGKNet(mn, νn)
p = init_params(nn)

nn(rand(Float32, vs.nu * 2 + 1, 2), p, vs, 5 / 3)

#--- class 2 ---#
# full distribution
mn = FnChain(FnDense(vs.nu, vs.nu, tanh; bias = false), FnDense(vs.nu, 3; bias = false))
νn = FnChain(
    FnDense(vs.nu + 1, vs.nu, tanh; bias = false),
    FnDense(vs.nu, vs.nu, relu; bias = false),
)
nn = BGKNet(mn, νn)
p = init_params(nn)

nn(rand(Float32, vs.nu + 1, 2), p, vs, 3, Class{2})

# reduced distribution
mn = FnChain(
    FnDense(vs.nu * 2, vs.nu * 2, tanh; bias = false),
    FnDense(vs.nu * 2, 6; bias = false),
)
νn = FnChain(
    FnDense(vs.nu * 2 + 1, vs.nu * 2 + 1, tanh; bias = false),
    FnDense(vs.nu * 2 + 1, vs.nu * 2, relu; bias = false),
)
nn = BGKNet(mn, νn)
p = init_params(nn)

nn(rand(Float32, vs.nu * 2 + 1, 2), p, vs, 5 / 3, Class{2})
