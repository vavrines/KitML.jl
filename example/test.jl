include("math.jl")
using KitBase


quadratureorder =2
points, triangulation = KitBase.octa_quadrature(quadratureorder)

println("-------------------")
#println(points)
println(size(points))


#nq = size(points, 1)

momentBasis =  ComputeSphericalBasisKarth(points,2, 3)
#momentBasis2 = ComputeSphericalBasisAnalytical(points)
println(size(momentBasis))
println(momentBasis)
#println(size(momentBasis2))
#println(momentBasis2-momentBasis)

#create points
points1D = linspace(-1,1,5)
points1DM = zeros(5,1)
points1DM[:,1] = points1D[:]
#println(size(points1DM))


momentBasis =  ComputeSphericalBasisKarth(points1DM,2,1)
momentBasis2 =  ComputeSphericalBasisAnalytical(points1DM)
println(size(momentBasis))
println(size(momentBasis2))
println(momentBasis-momentBasis2)
