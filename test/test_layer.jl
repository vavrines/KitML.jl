vs = KitML.KitBase.VSpace1D(-5, 5, 16; precision = Float32)

mn = FnChain(FnDense(vs.nu, vs.nu, tanh; bias = false), FnDense(vs.nu, vs.nu; bias = false))
νn = FnChain(FnDense(vs.nu, vs.nu, tanh; bias = false), FnDense(vs.nu, vs.nu, sigmoid; bias = false))
nn = BGKNet(mn, νn)
p = init_params(nn)

nn(rand(Float32, vs.nu, 2), p, vs)
