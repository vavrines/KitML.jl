cd(@__DIR__)
KitML.load_data("dataset.csv", dlm = ",")

nn1 = KitML.load_model("jmodel.jld2")
nn2 = KitML.load_model("tfmodel.h5")

KitML.save_model(nn1)
KitML.save_model(nn2; mode = :tf)
