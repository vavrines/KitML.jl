# ============================================================
# I / O Methods
# ============================================================

export save_model

"""
    save_model(nn; mode = :jld)

Save the trained machine learning model
"""
function save_model(nn; mode = :jld)
    if mode == :jld
        @save "model.jld2" nn
    end
end
