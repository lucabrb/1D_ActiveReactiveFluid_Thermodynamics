# Packages
using LinearAlgebra
using Plots
using JLD
using NLsolve
using Roots
using Symbolics
cd(@__DIR__)

# Model parameters
k  = [1.0 1.0 1.0 1.0 1.0 1.0] #k1, k2, ... k6
k_ = [1.0 0.0 0.0 0.0 0.0 0.0] #k-1, k-2, ... k-6
D  = [0.1 1.0 0.1 1.0 0.01 1.0] #D_na, D_ni, D_gm, D_gc, D_cm, D_cc
n = 1.0
g = 10.0
c = 1.0

DD = zeros(Float64, 6, 6)

# We produce a cut of the stability diagram in the plane spanned by two control parameters, Par1 Par2
# Par1 ∈ [MinPar1, MaxPar1]
MinPar1 = 0
MaxPar1 = 50
# Par2 ∈ [MinPar2, MaxPar2]
MinPar2 = 0
MaxPar2 = 50
NPar1 = 50 # n. of points in the grid of Par1
NPar2 = 50 # n. of points in the grid of Par2
ΔPar1 = (MaxPar1 - MinPar1) / NPar1 # grid spacing for Par1
ΔPar2 = (MaxPar2 - MinPar2) / NPar2 # grid spacing for Par2

# Stability Diagram is a (NPar1 + 1) times (NPar2 + 1) matrix, whose entries will be:
# 0, if HSS is stable
# 1, if HSS is unstable and fastest growing eigenvalue is real
# 2, if HSS is unstable and fastest growing eigenvalue is complex
StabilityDiagram = zeros(Int64, NPar1 + 1, NPar2 + 1)

# Initialize real axis (prerequisite to initialize reciprocal axis)
L = 5*π
Nx = 256
Δx = L / (Nx - 1)
# Initialize reciprocal axis
Kmax = π / Δx
Kmin = 2*π / L
Nk = Int64(floor((Kmax - Kmin) / Kmin))

StabilityMatrix = zeros(Float64, 6, 6) # This is the matrix M mentioned in the Supplemental Material
EigenvStabilityMatrix = zeros(Complex{Float64}, 6, Nk + 1) # Matrix containing the three eigenvalues of M (lines), as functions of K (columns)

gm_aux = zeros(Float64, 2)
HSS    = zeros(Float64, 3)
# This function produces StabilityDiagram
function StabilityMatrix!(gm_aux, HSS, k, k_, D, DD, n, g, c, NPar1, MinPar1, ΔPar1, NPar2, MinPar2, ΔPar2, Nk, EigenvStabilityMatrix, StabilityDiagram)
    for (i, j) in collect(Iterators.product(1:NPar1+1, 1:NPar2+1))
        k[4] = MinPar1 + (i - 1) * ΔPar1
        k[5] = MinPar2 + (j - 1) * ΔPar2
        #Find HSS
        gm_eq(x) = ((c*((n*(k[2]*x + k_[1])*k[3]) / (k[2]*x + k_[2]*x + k[1] + k_[1]) + k_[6])*k_[5]) / ((n*(k[2]*x + k_[1])*k[3]) / (k[2]*x + k_[2]*x + k[1] + k_[1]) + (n*(k[2]*x + k_[1])*k_[3]) / (k[2]*x + k_[2]*x + k[1] + k_[1]) + k[6] + k_[6]) + (n*(k[2]*x + k_[1])*k[4]) / (k[2]*x + k_[2]*x + k[1] + k_[1]))*(g - x) + ((-c*((n*(k[2]*x + k_[1])*k[3]) / (k[2]*x + k_[2]*x + k[1] + k_[1]) + k_[6])*k[5]) / ((n*(k[2]*x + k_[1])*k[3]) / (k[2]*x + k_[2]*x + k[1] + k_[1]) + (n*(k[2]*x + k_[1])*k_[3]) / (k[2]*x + k_[2]*x + k[1] + k_[1]) + k[6] + k_[6]) + (-n*(k[2]*x + k_[1])*k_[4]) / (k[2]*x + k_[2]*x + k[1] + k_[1]))*x
        gm_aux[1] = find_zero(gm_eq, (g/2))
        gm_aux[2] = find_zero(gm_eq, (0,g))
        HSS[2] = maximum(gm_aux)
        HSS[1] = (n*(k[2]*HSS[2] + k_[1])) / (k[1] + k_[1] + k[2]*HSS[2] + k_[2]*HSS[2])
        HSS[3] = (c*(k[3]*HSS[1] + k_[6])) / (k[6] + k_[6] + k[3]*HSS[1] + k_[3]*HSS[1])
        println(HSS)
        # ∂_t x = D ∂_xx x + R(x) ~ M x; with M = - D q^2 + ∇R
        # Find ∇R 
        @variables y[1:6]
        function R(y)
            [
            - y[1] * (k[1] + k_[2] * y[3]) + y[2] * (k_[1] + k[2] * y[3]),
              y[1] * (k[1] + k_[2] * y[3]) - y[2] * (k_[1] + k[2] * y[3]),
            - y[3] * (k_[4] * y[1] + k[5] * y[5]) + y[4] * (k[4] * y[1] + k_[5] * y[5]),
              y[3] * (k_[4] * y[1] + k[5] * y[5]) - y[4] * (k[4] * y[1] + k_[5] * y[5]),
            - y[5] * (k[6] + k_[3] * y[1]) + y[6] * (k_[6] + y[1] * k[3]),
              y[5] * (k[6] + k_[3] * y[1]) - y[6] * (k_[6] + y[1] * k[3])]
        end
        ∇R = Symbolics.jacobian(R(y), y[1:6])
        ∇R_eval = Float64.(Symbolics.value.(
                            substitute.(∇R, (Dict(
                                y[1] => HSS[1], 
                                y[2] => n - HSS[1],
                                y[3] => HSS[2], 
                                y[4] => g - HSS[2],
                                y[5] => HSS[3], 
                                y[6] => c - HSS[3]),))))
        # Initialize DD matrix
        for i in 1:6
            DD[i,i] = D[i]
        end
        # Find eigenvalues of StabilityMatrix at current parameters
        for p = 1:Nk
            K = p * Kmin
            StabilityMatrix = - (K^2) * DD + ∇R_eval
            EigenvStabilityMatrix[:, p] = eigvals(StabilityMatrix)
        end
        # Find eigenvalue with largest real part MaxEigen = MaxEigenvRe + im * MaxEigenvIm
        MaxEigenvIndex = findmax(real.(EigenvStabilityMatrix))[2] # MaxEigen is the element of StabilityMatrix indexed MaxEigenvIndex[1],MaxEigenvIndex[2]
        MaxEigenvRe = findmax(real.(EigenvStabilityMatrix))[1]
        MaxEigenvIm = imag.(EigenvStabilityMatrix[MaxEigenvIndex[1], MaxEigenvIndex[2]])
        # Entries of StabilityDiagram
        if MaxEigenvRe > 0
            if MaxEigenvIm == 0
                StabilityDiagram[i, j] = 1 # Signals Turing bifurcation
            else
                StabilityDiagram[i, j] = 2 # Signals Hopf bifurcation
            end
        else
            StabilityDiagram[i, j] = 0 # Homogeneous steady state is stable
        end
    end
end

# This function calculates the Turing- and Hopf- stability lines in parameter space (TuringLine and HopfLine, respectively)
function BifurcationLines(NPar1, MinPar1, ΔPar1, NPar2, MinPar2, ΔPar2)
    TuringLine = [0,0]
    HopfLine = [0,0]
    for (i, j) in collect(Iterators.product(1:NPar1+1, 1:NPar2+1))
        P1 = MinPar1 + (i - 1) * ΔPar1
        P2 = MinPar2 + (j - 1) * ΔPar2
        # The condition below locates a Turing bifurcation in parameter space, by looking for transitions from 0 to 1 in the entries of StabilityDiagram
        if (i>1 && StabilityDiagram[i,j] == 1 && StabilityDiagram[i-1,j] == 0) || (j>1 && StabilityDiagram[i,j] == 0 && StabilityDiagram[i,j-1] == 1)
            TuringLine = hcat(TuringLine, [P1, P2])
        end
        # The condition below locates a Hopf bifurcation in parameter space, by looking for transitions from 0 to 2 in the entries of StabilityDiagram
        if (i>1 && StabilityDiagram[i,j] == 2 && StabilityDiagram[i-1,j] == 0) || (j>1 && StabilityDiagram[i,j] == 0 && StabilityDiagram[i,j-1] == 2)
            HopfLine = hcat(HopfLine, [P1, P2])
        end
    end
    TuringLine = transpose(TuringLine)
    HopfLine = transpose(HopfLine)
    return TuringLine[2:end,:], HopfLine[2:end,:]
end

# Calculate instability lines and plot stability diagram
StabilityMatrix!(gm_aux, HSS, k, k_, D, DD, n, g, c, NPar1, MinPar1, ΔPar1, NPar2, MinPar2, ΔPar2, Nk, EigenvStabilityMatrix, StabilityDiagram)
TuringLine, HopfLine = BifurcationLines(NPar1, MinPar1, ΔPar1, NPar2, MinPar2, ΔPar2)
p = scatter(TuringLine[:,1], TuringLine[:,2],
    xlabel = "Parameter 1",
    ylabel = "Parameter 2",
    label = "Turing")
p = scatter!(HopfLine[:,1], HopfLine[:,2],
    label = "Hopf")

#= @variables x[1:3], k[1:6], k_[1:6], n, g, c

na_hss = (n*(k[2]*x[2] + k_[1])) / (k[1] + k_[1] + k[2]*x[2] + k_[2]*x[2])
cm_hss = (c*(k[3]*x[1] + k_[6])) / (k[6] + k_[6] + k[3]*x[1] + k_[3]*x[1])
cm_hss = substitute.(cm_hss, (Dict(
                                x[1] => na_hss),))[1]

gm_eq = - x[2] * (k_[4] * x[1] + k[5] * x[3]) + (g - x[2]) * (k[4] * x[1] + k_[5] * x[3])
gm_eq = substitute.(gm_eq, (Dict(
                    x[1] => na_hss, 
                    x[3] => cm_hss),))[1] =#