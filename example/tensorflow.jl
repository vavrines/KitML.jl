import KitML

# load tensorflow model
cd(@__DIR__)
model = KitML.load_model("best_model.h5"; mode = :tf)
model.summary()

# prepare data
X = randn(Float32, 10, 4)
Y = randn(Float32, 10, 1)

# reinforcement training
KitML.sci_train!(model, (X, Y); epoch = 2)
# it's equivalent as
model.fit(X, Y, epochs=2)

model.predict(X)
