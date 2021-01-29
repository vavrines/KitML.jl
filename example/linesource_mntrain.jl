using Flux
import KitML

# dataset
cd(@__DIR__)
dataset = KitML.load_data("data_mn.csv"; mode = :csv, dlm = ",")

baseDimension = 4 #length of u

for name in names(dataset)
    println(":", name, ", ")
end
 

L = size(dataset, 1)

# surrogate 1: moments -> entropy
begin
    X = zeros(baseDimension, size(dataset, 1))
    Y = zeros(1, size(dataset, 1))
    for j in axes(X, 2)
        X[1, j] = dataset.u_0[j];;
        X[2, j] = dataset.u_1[j]
        X[3, j] = dataset.u_2[j]
        X[4, j] = dataset.u_3[j]
        Y[1, j] = dataset.h[j]
    end
    data = Flux.Data.DataLoader(X, Y, batchsize = 256, shuffle=true)
    
    #Base.show(nn)
    #nn = Chain(
    #    Dense(4, 16, relu),
    #    Dense(16, 16, relu),
    #    Dense(16, 16, relu),
    #    Dense(16, 1),
    #)
    nn = KitML.create_neural_closure(baseDimension, 1, 8, 3)


end
KitML.sci_train!(nn, data, ADAM(), 1000)
KitML.save_model(nn)
println("End ICNN Training")

# surrogate 2: ResNet
nn2 = Chain(
    Dense(4, 16, relu),
    KitML.Shortcut(Chain(Dense(16, 16, relu), Dense(16, 16, relu))),
    Dense(16, 1),
)
KitML.sci_train!(nn2, data, ADAM(), 2)

"""Notice the training of mapping u to Lagrangian multiplier λ seems not working well"""
