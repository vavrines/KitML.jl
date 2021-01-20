# ============================================================
# I / O Methods
# ============================================================

export load_data,
       save_model

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
    save_model(nn; mode = :jld)

Save the trained machine learning model
"""
function save_model(nn; mode = :jld)
    if mode == :jld
        @save "model.jld2" nn
    end
end
