cd(@__DIR__)
KitML.load_data("dataset.csv", dlm = ",")

nn1 = KitML.load_model("jmodel.jld2")
nn2 = KitML.load_model("tfmodel.h5")
#nn1 = KitML.load_model("jmodel.jld2"; mode = :jld)
#nn2 = KitML.load_model("tfmodel.h5"; mode = :tf)

KitML.save_model(nn1)
KitML.save_model(nn2; mode = :tf)
