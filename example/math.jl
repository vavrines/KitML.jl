########################
maxwell_boltzmann_primal(f) = f.* log.(f) - f 

function computeEntropyH(α, m, ω,η::Function, η_prime_dual::Function)
    result = sum(η(η_prime_dual.(α' * m)[:] ).* ω )
    return result
end

#############################################


function ComputeSphericalBasisKarth(quadpts::Matrix{Float64}, polyDegree::Int64, spatialDim::Int64)
    monomialBasis = zeros(GetBasisSize(polyDegree,spatialDim),size(quadpts,1))
    
    if spatialDim == 3
        for idx_quad in 1:(size(quadpts)[1])
            monomialBasis[:,idx_quad]  = ComputeShpericalBasis3DPtKarth(quadpts[idx_quad,1],quadpts[idx_quad,2],quadpts[idx_quad,3],polyDegree)
        end
    elseif spatialDim == 1
        for idx_quad in 1:(size(quadpts)[1])
            monomialBasis[:,idx_quad]= ComputeShpericalBasis1DPtKarth(quadpts[idx_quad,1],polyDegree)
        end
    end

    return monomialBasis

end

function ComputeShpericalBasis3DPtKarth(pointX::Float64,pointY::Float64,pointZ::Float64,polyDegree::Int64)
    idx_vector = 1
    spatialDim = 3
    basisLen = GetBasisSize(polyDegree,spatialDim)
    basisAtPt = ones(1,basisLen)
    for idx_degree in 0:polyDegree
        for a in 0:idx_degree
            for b in 0:(idx_degree-a)
                c = idx_degree - a - b
                basisAtPt[idx_vector] = Power(pointX,a)*Power(pointY,b)*Power(pointZ,c)
                idx_vector = idx_vector+1
            end
        end
    end
    
    return basisAtPt
end

function ComputeShpericalBasis1DPtKarth(pointX::Float64,polyDegree::Int64)
    idx_vector = 1
    spatialDim = 1
    basisLen = GetBasisSize(polyDegree,spatialDim)
    basisAtPt = ones(1,basisLen)
    for a in 0:polyDegree
                basisAtPt[idx_vector] = Power(pointX,a)
                idx_vector = idx_vector+1
    end
    
    return basisAtPt
end

function ComputeSphericalBasisAnalytical(quadpts::Matrix{Float64})
    # Hardcoded solution for L = 2, spatialDim = 3
    
    #monomialBasis = zeros(10,size(quadpts,1))
    #for idx_quad in 1:(size(quadpts)[1])
    #    monomialBasis[1,idx_quad]  = 1
    #    monomialBasis[2,idx_quad]  = quadpts[idx_quad,3] # z
    #    monomialBasis[3,idx_quad]  = quadpts[idx_quad,2] # y
    #    monomialBasis[4,idx_quad]  = quadpts[idx_quad,1] # x 
    #    monomialBasis[5,idx_quad]  = quadpts[idx_quad,3] * quadpts[idx_quad,3] # z*z
    #    monomialBasis[6,idx_quad]  = quadpts[idx_quad,2] * quadpts[idx_quad,3] # y*z
    #    monomialBasis[7,idx_quad]  = quadpts[idx_quad,2] * quadpts[idx_quad,2] # y*y 
    #    monomialBasis[8,idx_quad]  = quadpts[idx_quad,1] * quadpts[idx_quad,3] # x*z 
    #    monomialBasis[9,idx_quad]  = quadpts[idx_quad,1] * quadpts[idx_quad,2] # x*y 
    #    monomialBasis[10,idx_quad] = quadpts[idx_quad,1] * quadpts[idx_quad,1] # x*x
    #end

    monomialBasis = zeros(3,size(quadpts,1))
    for idx_quad in 1:(size(quadpts)[1])
        monomialBasis[1,idx_quad]  = 1
        monomialBasis[2,idx_quad]  = quadpts[idx_quad,1] # z
        monomialBasis[3,idx_quad]  = quadpts[idx_quad,1] * quadpts[idx_quad,1] # z*z
    end

    return monomialBasis
end

function Power(  basis,  exponent ) 
    if exponent == 0 
        return 1.0
    end
    result = basis;
    for i in 2:exponent
        result = result * basis
    end
    return result
end

function GetBasisSize(LMaxDegree, spatialDim)
    basisLen = 0
    for idx_degree in 0:LMaxDegree
        basisLen = basisLen + GetCurrDegreeSize(idx_degree,spatialDim)
    end
    return convert(Int,basisLen)
end

function GetCurrDegreeSize(currDegree, spatialDim)
    return factorial(currDegree + spatialDim -1)/(factorial(currDegree)*factorial(spatialDim-1 ))
end