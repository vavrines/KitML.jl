using Flux
import KitML

a = KitML.ICNNLayer(5, 5, 5, tanh)
c = KitML.ICNNChain(5,5,[5,5])
c(randn(5))
