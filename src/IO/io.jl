# ============================================================
# I / O Methods
# ============================================================

export load_data, load_model, save_model

"""
    load_data(file; mode = :csv, dlm = " ")

Load dataset from file
"""
function load_data(file; mode = :csv, dlm = nothing)
    if mode == :csv
        dataset = CSV.File(file; delim = dlm) |> DataFrame
    end

    return dataset
end

"""
    load_model(file; mode = :jld)

Load the trained machine learning model
"""
function load_model(file::T; mode = :jld) where {T<:AbstractString}
    if mode == :jld
        @load file nn
    elseif mode == :tf
        tf = pyimport("tensorflow")
        nn = tf.keras.models.load_model(file)
    end

    return nn
end


"""
    save_model(nn; mode = :jld)

Save the trained machine learning model
"""
function save_model(nn; mode = :jld)
    if mode == :jld
        @save "model.jld2" nn
    elseif mode == :tf
        tf = pyimport("tensorflow")
        nn.save("model.h5")
    end
end
