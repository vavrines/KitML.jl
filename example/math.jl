
function ComputeSphericalBasis(LMaxDegree,spatialDim,quadpts)
    my = 0
    phi = 0
    monomialBasis = zeros(GetBasisSize(LMaxDegree,spatialDim),size(quadpts,1))
    for idx_quad in 0:(size(quadpts)[1]-1)
        ptsSphere = transformToSphere(quadpts[idx_quad+1,:])

        my = ptsSphere[1]
        phi = ptsSphere[2]
        monomialBasis[:,idx_quad+1]  = ComputeShpericalBasis3DPt(my,phi,LMaxDegree)
    end

    return monomialBasis
end

function ComputeSphericalBasisAnalytical(L,spatialDim,quadpts)
    # Hardcoded solution for L = 1, spatialDim = 3
    
    monomialBasis = zeros(4,size(quadpts,1))
    for idx_quad in 0:(size(quadpts)[1]-1)
        monomialBasis[1,idx_quad+1]  = 1
        monomialBasis[2,idx_quad+1]  = quadpts[idx_quad+1,1] # x
        monomialBasis[3,idx_quad+1]  = quadpts[idx_quad+1,2] # y
        monomialBasis[4,idx_quad+1]  = quadpts[idx_quad+1,3] # z 
    end

    return monomialBasis
end

function ComputeShpericalBasis3DPt(my, phi, LMaxDegree)
    spatialDim = 3
    idx_vector = 0
    omegaX = 0
    omegaY = 0
    omegaZ = 0
    a = 0
    b = 0
    c = 0
    basisLen = GetBasisSize(LMaxDegree,spatialDim)
    basisAtPt = ones(1,basisLen)

    for idx_degree in 0:LMaxDegree
        omegaX = Omega_x( my, phi )
        omegaY = Omega_y( my, phi )
        omegaZ = Omega_z( my ) 

        for a in 0:idx_degree
            for b in 0:(idx_degree-a)
                c = idx_degree - a - b
                basisAtPt[idx_vector+1] = Power(omegaX,a)*Power(omegaY,b)*Power(omegaZ,c)
                idx_vector = idx_vector+1
            end
        end
    end
    
    return basisAtPt
end

# Helper Functions
function Omega_x( my, phi ) 
    return sqrt( 1 - my * my ) * sin( phi )
end

function Omega_y( my, phi ) 
    return sqrt( 1 - my * my ) * cos( phi )
end

function Omega_z( my ) 
    return my
end

function Power(  basis,  exponent ) 
    if exponent == 0 
        return 1.0
    end
    result = basis;
    for i in 1:exponent
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

function transformToSphere(pointsKarthesian)
    my = pointsKarthesian[3]
    phi = 0
    if (pointsKarthesian[2]>0)
        phi = acos(pointsKarthesian[1])
    else
        phi = 2*π-acos(pointsKarthesian[1])
    end    
    return [my,phi]
end