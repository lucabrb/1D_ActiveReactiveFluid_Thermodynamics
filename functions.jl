function CalculateRHS!(k, k_, D, v, Fields, ∂xx, ∂xFieldsv, σ, RHS, q, P, Pinv, Factor1stDer, Factor2ndDer)
    # This funct. calculates RHS of dynamical equations, written as ∂t (density) = RHS
    # Calculate 2nd derivatives w/ FFTs
    for i in 1:6
        ∂xx[i,:] .= Pinv * ((P * Fields[i,:]) .* Factor2ndDer)
    end
    # Calculate velocity
    #σ[:] .= @. Pars["Z"] * (Fields[1,:] ^ 2) - Pars["B"] * (Fields[1,:] ^ 3)   # Calculate stress
    #v[:] .= Pinv * ( (P * σ) .* (im .* q) ./ (1 .+ q.*q) )                     # Calculate velocity w/ FFTs
    # Calculate advective currents w/ FFTs
    #for i in 1:6
    #    ∂xFieldsv[i,:] = Pinv * ((P * (Fields[i,:] .* v)) .* Factor1stDer)
    #end
    ## Calculate right hand sides
    # Starting w/ chemical reactions
    @. RHS[1,:] = - Fields[1,:] * (k[1] + k_[2] * Fields[3,:]) + Fields[2,:] * (k_[1] + k[2] * Fields[3,:])
    @. RHS[2,:] = - RHS[1,:]
    @. RHS[3,:] = Fields[1,:] * (k[4] * Fields[4,:] - k_[4] * Fields[3,:]) + Fields[5,:] * (- k[5] * Fields[3,:] + k_[5] * Fields[4,:])
    @. RHS[4,:] = - RHS[3,:]
    @. RHS[5,:] = Fields[1,:] * (k[3] * Fields[6,:] - k_[3] * Fields[5,:]) - k[6] * Fields[5,:] + k_[6] * Fields[6,:]
    @. RHS[6,:] = - RHS[5,:]
    # Adding diffusion
    for i in 1:6
        @. RHS[i,:] += D[i] * ∂xx[i,:]
    end
end

function EulerForward!(Δt, k, k_, D, v, Fields, ∂xx, ∂xFieldsv, σ, RHS, q, P, Pinv, Factor1stDer, Factor2ndDer)
    # This funct. propagates dynamical fields in time using Euler forward
    # Calculate right hand sides
    CalculateRHS!(k, k_, D, v, Fields, ∂xx, ∂xFieldsv, σ, RHS, q, P, Pinv, Factor1stDer, Factor2ndDer)
    # Update fields with their values at t + Δt
    @. Fields[:,:] += Δt * RHS[:,:]
    return nothing
end

function MidpointMethod!(Δt, k, k_, D, v, Fields, Fields_Aux, ∂xx, ∂xFieldsv, σ, RHS, q, P, Pinv, Factor1stDer, Factor2ndDer)
    # This funct. propagates dynamical fields in time using the midpoint method
    # Save fields at step t
    Fields_Aux[:,:] .= Fields[:,:]
    # Update fields with their values at midpoint (i.e., at t + Δt/2)
    EulerForward!(0.5*Δt, k, k_, D, v, Fields, ∂xx, ∂xFieldsv, σ, RHS, q, P, Pinv, Factor1stDer, Factor2ndDer)
    # Calculate RHS at midpoint by using updated fields
    CalculateRHS!(k, k_, D, v, Fields, ∂xx, ∂xFieldsv, σ, RHS, q, P, Pinv, Factor1stDer, Factor2ndDer)
    # Update fields with their values at t + Δt
    Fields[:,:] .= Fields_Aux[:,:] + Δt * RHS[:,:]
    return nothing
end

function AdaptiveTimeStep!(ErrorTolerance, ΔtVec, k, k_, D, v, Fields, Fields_Aux, v_LongStep, Fields_LongStep, v_ShortStepAux, Fields_ShortStepAux, v_ShortStep, Fields_ShortStep, ∂xx, ∂xFieldsv, σ, RHS, q, P, Pinv, Factor1stDer, Factor2ndDer)
    # First run
    # Propagate fields once by Δt
    Fields_LongStep[:,:] .= Fields[:,:]
    MidpointMethod!(ΔtVec[2], k, k_, D, v_LongStep, Fields_LongStep, Fields_Aux, ∂xx, ∂xFieldsv, σ, RHS, q, P, Pinv, Factor1stDer, Factor2ndDer)
    # Propagate fields twice by 0.5*Δt, first integration
    Fields_ShortStep[:,:] .= Fields[:,:]
    MidpointMethod!(0.5*ΔtVec[2], k, k_, D, v_ShortStep, Fields_ShortStep, Fields_Aux, ∂xx, ∂xFieldsv, σ, RHS, q, P, Pinv, Factor1stDer, Factor2ndDer)
    # Save current half step fields for next run
    v_ShortStepAux[:]  .= v_ShortStep
    Fields_ShortStepAux[:,:] .= Fields_ShortStep[:,:]
    # Propagate fields twice by 0.5*Δt, second integration
    MidpointMethod!(0.5*ΔtVec[2], k, k_, D, v_ShortStep, Fields_ShortStep, Fields_Aux, ∂xx, ∂xFieldsv, σ, RHS, q, P, Pinv, Factor1stDer, Factor2ndDer)
    # Calculate maximum relative error
    MaxError = findmax(abs.((Fields_LongStep .- Fields_ShortStep) ./ Fields_LongStep))[1]
    # Proceed by halving Δt until MaxError < ErrorTolerance
    while MaxError >= ErrorTolerance
        # Halven time step
        ΔtVec[2] = 0.5*ΔtVec[2]
        # LongStep is one half step in previous run (i.e. "Aux" fields)
        v_LongStep[:]  .= v_ShortStepAux
        Fields_LongStep[:,:] .= Fields_ShortStepAux[:,:]
        # Propagate fields twice by 0.5*Δt, first integration
        Fields_ShortStep[:,:] .= Fields[:,:]
        MidpointMethod!(0.5*ΔtVec[2], k, k_, D, v_ShortStep, Fields_ShortStep, Fields_Aux, ∂xx, ∂xFieldsv, σ, RHS, q, P, Pinv, Factor1stDer, Factor2ndDer)
        # Save current half step fields for next run
        v_ShortStepAux[:]  .= v_ShortStep
        Fields_ShortStepAux[:,:] .= Fields_ShortStep[:,:]
        # Propagate fields twice by 0.5*Δt, second integration
        MidpointMethod!(0.5*ΔtVec[2], k, k_, D, v_ShortStep, Fields_ShortStep, Fields_Aux, ∂xx, ∂xFieldsv, σ, RHS, q, P, Pinv, Factor1stDer, Factor2ndDer)
        # Calculate maximum relative error
        MaxError = findmax(abs.(Fields_LongStep .- Fields_ShortStep) ./ Fields_LongStep)[1]
    end
    # Update Δt_old
    ΔtVec[1] = ΔtVec[2]
    # Update fields
    Fields[:,:] .= Fields_LongStep[:,:]
    v[:]  .= v_LongStep
    return nothing
end
