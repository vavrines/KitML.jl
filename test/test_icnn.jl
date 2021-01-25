using Flux
import KitML

a = KitML.ICNNLayer(5, 5, 5, tanh)

a(rand(5), rand(5))