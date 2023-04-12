# All quantities are expressed in non-dimensional units
# Space grid
const L = 5*π                   # System size
const Nx = 256                  # Number of grid nodes
const Δx = L / (Nx-1)           # Grid spacing
x = [-L/2 + i*Δx for i = 0:Nx-1]   # Space vector

# Model parameters (saved in a dictionary)
k  = [1 1 1 1 1 1] #k1, k2, ... k6
k_ = [0 0 0 0 0 0] #k1_, k2_, ... k6_
D  = [1 1 1 1 1 1] #D_na, D_ni, D_gm, D_gc, D_cm, D_cc
const n = 1
const g = 1
const c = 1

# Homogeneous Steady State (used in initial conditions below)
# Solving EqnHSS = 0 for Na gives Na at HSS

function HSS_system!(F, x)
    F[1] = - x[1] * (1 + k_[2] * x[2]) + (1 - x[1]) * (k_[1] + k[2] * x[2])
    F[2] = x[1] * (k[4] * (1-x[2]) - k_[4] * x[2]) + x[3] * (- k[5] * x[2] + k_[5] * (1-x[2]))
    F[3] = x[1] * (k[3] * (1-x[3]) - k_[3] * x[3]) - k[6] * x[3] + k_[6] * (1-x[3])
end

HSS = nlsolve(HSS_system!, [n/2 g/2 c/2]).zero

# Initial Conditions (HSS + weak noise)
FieldICs = zeros(Float64, 6, Nx)
# Parameters of noisy IC
seedd = 123                                     # Seed of rand. number generator
Random.seed!(seedd)                             # Seeds random number generator
ε   = 0.5                                      # Noise amplitude, small number
Noise = ε .* (-1 .+ 2 .* rand(Float64, Nx))     # Nx-long vector of weak noise, made of random numbers between ε*[-1, 1]
ZeroMeanNoise = Noise .- integrate(x, Noise)/L  # Noisy vector with zero average, s.t. HSS + Noise conserves tot. number of molecules
# Uncomment if desired IC is HSS + weak noise
@. FieldICs[1,:] = HSS[1]       * (1 + ZeroMeanNoise)
@. FieldICs[2,:] = (n - HSS[1]) * (1 + ZeroMeanNoise)
@. FieldICs[3,:] = HSS[2]       * (1 + ZeroMeanNoise)
@. FieldICs[4,:] = (g - HSS[2]) * (1 + ZeroMeanNoise)
@. FieldICs[5,:] = HSS[3]       * (1 + ZeroMeanNoise)
@. FieldICs[6,:] = (c - HSS[3]) * (1 + ZeroMeanNoise)

# Final time reached by simulation
FinalTime = 10

# Adaptive timestep parameters
MinTimeStep = 1e-15         # Simulation stops if time step drops below MinTimeStep
ErrorTolerance = 1e-10      # Error tolerance between Δt and 0.5*Δt steps

# Saving parameters
# Output file parameters
FrameTimeStep = 0.1                                  # Save state of system with time intervals = FrameTimeStep (in non-dim. units)
plotframes = floor(Int64, FinalTime / FrameTimeStep) # Total number of system states saved during the simulation
dt = Dates.format(now(), "yyyymmdd")                 # String of today's date
DataFileName = "Data/"*dt*"-Data.jld"                # Name of output file