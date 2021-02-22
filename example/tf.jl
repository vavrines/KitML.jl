using PyCall

tf = pyimport("tensorflow")

cd(@__DIR__)
model = tf.keras.models.load_model("best_model.h5")

x = randn(Float32, 1, 4)
model.predict(x)