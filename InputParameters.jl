# All quantities are expressed in non-dimensional units
# Space grid
const L = 5*π                   # System size
const Nx = 256                  # Number of grid nodes
const Δx = L / (Nx-1)           # Grid spacing
x = [-L/2 + i*Δx for i = 0:Nx-1]   # Space vector

# Model parameters
k  = [0.1 1.0 10 10.0 2.0 1.0] #k1, k2, ... k6
k_ = [0.0 0.0 0.0 0.0 0.0 0.0] #k-1, k-2, ... k-6
D  = [0.1 1.0 0.1 1.0 0.01 1.0] #D_na, D_ni, D_gm, D_gc, D_cm, D_cc
n = 1.0
g = 1.0
c = 1.0

# Homogeneous Steady State (used in initial conditions below)
gm_aux = zeros(Float64, 2)
HSS    = zeros(Float64, 3)
#
gm_eq(x) = (g - x)*((-c*((-n*(k[2]*x + k_[1])*k[3]) / (-k[1] - k_[1] - k[2]*x - k_[2]*x) + k_[6])*k_[5]) / ((n*(k[2]*x + k_[1])*k[3]) / (-k[1] - k_[1] - k[2]*x - k_[2]*x) + (n*(k[2]*x + k_[1])*k_[3]) / (-k[1] - k_[1] - k[2]*x - k_[2]*x) - k[6] - k_[6]) + (-n*(k[2]*x + k_[1])*k[4]) / (-k[1] - k_[1] - k[2]*x - k_[2]*x)) + ((n*(k[2]*x + k_[1])*k_[4]) / (-k[1] - k_[1] - k[2]*x - k_[2]*x) + (c*((-n*(k[2]*x + k_[1])*k[3]) / (-k[1] - k_[1] - k[2]*x - k_[2]*x) + k_[6])*k[5]) / ((n*(k[2]*x + k_[1])*k[3]) / (-k[1] - k_[1] - k[2]*x - k_[2]*x) + (n*(k[2]*x + k_[1])*k_[3]) / (-k[1] - k_[1] - k[2]*x - k_[2]*x) - k[6] - k_[6]) - (n + (n*(k[2]*x + k_[1])) / (-k[1] - k_[1] - k[2]*x - k_[2]*x))*k[2])*x
gm_aux[1] = find_zero(gm_eq, (g/2))
gm_aux[2] = find_zero(gm_eq, (0,g))
HSS[2] = maximum(gm_aux)
HSS[1] = (n*(k[2]*HSS[2] + k_[1])) / (k[1] + k_[1] + k[2]*HSS[2] + k_[2]*HSS[2])
HSS[3] = (c*(k[3]*HSS[1] + k_[6])) / (k[6] + k_[6] + k[3]*HSS[1] + k_[3]*HSS[1])

# Initial Conditions (HSS + weak noise)
FieldICs = zeros(Float64, 6, Nx)
# Parameters of noisy IC
seedd = 123                                     # Seed of rand. number generator
Random.seed!(seedd)                             # Seeds random number generator
ε   = 0.01                                      # Noise amplitude, small number
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
FinalTime = 100

# Adaptive timestep parameters
MinTimeStep = 1e-15         # Simulation stops if time step drops below MinTimeStep
ErrorTolerance = 1e-10      # Error tolerance between Δt and 0.5*Δt steps

# Saving parameters
# Output file parameters
FrameTimeStep = 0.1                                  # Save state of system with time intervals = FrameTimeStep (in non-dim. units)
plotframes = floor(Int64, FinalTime / FrameTimeStep) # Total number of system states saved during the simulation
dt = Dates.format(now(), "yyyymmdd")                 # String of today's date
DataFileName = "Data/"*dt*"-Data.jld"                # Name of output file