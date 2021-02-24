using KitBase

cd(@__DIR__)
ks, ctr, face, t = initialize("config.txt")

dt = timestep(ks, ctr, t)
nt = Int(floor(ks.set.maxTime / dt))
res = zeros(3)






for iter = 1:nt
    reconstruct!(ks, ctr)
    evolve!(ks, ctr, face, dt)
    

end