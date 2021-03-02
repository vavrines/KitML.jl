cd(@__DIR__)
KitML.load_data("dataset.csv", dlm = ",")

nn1 = KitML.load_model("model.jld2", mode = :jld)
nn2 = KitML.load_model("tfmodel.h5"; mode = :tf)

KitML.save_model(nn1)
KitML.save_model(nn1)

using KitML.KitBase.PyCall
tf = pyimport("tensorflow")
nn = tf.keras.models.load_model("tfmodel.h5")