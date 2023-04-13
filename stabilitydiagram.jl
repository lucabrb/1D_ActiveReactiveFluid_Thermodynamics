# Packages
using LinearAlgebra
using Plots
using JLD
using NLsolve
using Symbolics
cd(@__DIR__)

# Model parameters
k  = [1.0 1.0 1.0 1.0 1.0 1.0] #k1, k2, ... k6
k_ = [0.0 0.0 0.0 0.0 0.0 0.0] #k-1, k-2, ... k-6
D  = [0.1 1.0 0.1 1.0 0.01 1.0] #D_na, D_ni, D_gm, D_gc, D_cm, D_cc
n = 10.0
g = 10.0
c = 1.0

DD = zeros(Float64, 6, 6)

# We produce a cut of the stability diagram in the plane spanned by two control parameters, Par1 Par2
# Par1 ∈ [MinPar1, MaxPar1]
MinPar1 = 0
MaxPar1 = 10
# Par2 ∈ [MinPar2, MaxPar2]
MinPar2 = 0
MaxPar2 = 10
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
L = 10*π
Nx = 512
Δx = L / (Nx - 1)
# Initialize reciprocal axis
Kmax = π / Δx
Kmin = 2*π / L
Nk = Int64(floor((Kmax - Kmin) / Kmin))

StabilityMatrix = zeros(Float64, 6, 6) # This is the matrix M mentioned in the Supplemental Material
EigenvStabilityMatrix = zeros(Complex{Float64}, 6, Nk + 1) # Matrix containing the three eigenvalues of M (lines), as functions of K (columns)

# This function produces StabilityDiagram
function StabilityMatrix!(k, k_, D, DD, n, g, c, NPar1, MinPar1, ΔPar1, NPar2, MinPar2, ΔPar2, Nk, EigenvStabilityMatrix, StabilityDiagram)
    count = 0
    for (i, j) in collect(Iterators.product(1:NPar1+1, 1:NPar2+1))
    count += 1
    println("Parameter space pt ", count, " of ", (NPar1+1)*(NPar2+1))
        k[4] = MinPar1 + (i - 1) * ΔPar1
        k[5] = MinPar2 + (j - 1) * ΔPar2
        #Find HSS
        function HSS_system!(F, x)
            F[1] = - x[1] * (1 + k_[2] * x[2]) + (1 - x[1]) * (k_[1] + k[2] * x[2])
            F[2] = x[1] * (k[4] * (1-x[2]) - k_[4] * x[2]) + x[3] * (- k[5] * x[2] + k_[5] * (1-x[2]))
            F[3] = x[1] * (k[3] * (1-x[3]) - k_[3] * x[3]) - k[6] * x[3] + k_[6] * (1-x[3])
        end
        HSS = nlsolve(HSS_system!, [n/2 g/2 c/2]).zero
        # ∂_t x = D ∂_xx x + R(x) ~ M x; with M = - D q^2 + ∇R
        # Find ∇R 
        @variables x[1:6]
        function R(x)
            [- x[1] * (1 + k_[2] * x[3]) + x[2] * (k_[1] + k[2] * x[3]),
            -(- x[1] * (1 + k_[2] * x[3]) + x[2] * (k_[1] + k[2] * x[3])),
            x[1] * (k[4] * x[4] - k_[4] * x[3]) + x[5] * (- k[5] * x[3] + k_[5] * x[4]),
            -(x[1] * (k[4] * x[4] - k_[4] * x[3]) + x[5] * (- k[5] * x[3] + k_[5] * x[4])),
            x[1] * (k[3] * x[6] - k_[3] * x[5]) - k[6] * x[5] + k_[6] * x[6],
            -(x[1] * (k[3] * x[6] - k_[3] * x[5]) - k[6] * x[5] + k_[6] * x[6])]
        end
        ∇R = Symbolics.jacobian(R(x), x[1:6])
        ∇R_eval = Float64.(Symbolics.value.(
                            substitute.(∇R, (Dict(
                                x[1] => HSS[1], 
                                x[2] => n - HSS[1],
                                x[3] => HSS[2], 
                                x[4] => g - HSS[2],
                                x[5] => HSS[3], 
                                x[6] => c - HSS[3]),))))
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
StabilityMatrix!(k, k_, D, DD, n, g, c, NPar1, MinPar1, ΔPar1, NPar2, MinPar2, ΔPar2, Nk, EigenvStabilityMatrix, StabilityDiagram)
TuringLine, HopfLine = BifurcationLines(NPar1, MinPar1, ΔPar1, NPar2, MinPar2, ΔPar2)
p = scatter(TuringLine[:,1], TuringLine[:,2],
    xlabel = "P2", ylabel = "P1",
    xlims = (MinPar2, MaxPar2),
    ylims = (MinPar1, MaxPar1),
    label = "Turing")
p = scatter!(HopfLine[:,1], HopfLine[:,2],
    label = "Hopf")