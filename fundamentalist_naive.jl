using DynamicalSystems, Plots
using Pkg
Pkg.add("Plots")
using LaTeXStrings
function generateinstance(a, p₀, p₁, p₂; R = 1.01)
    @inline @inbounds function evolution!(nextstate, state, parameters, t)
        p_t1, p_t2, p_t3 = state
        β, g, b = parameters
        prof1 = (p_t1 - R*p_t2)*(g*p_t3 - R*p_t2)
        prof2 = (p_t1 - R*p_t2)*(b -R*p_t2)
        num1 = exp(β*(prof1))
        num2 = exp(β*(prof2))
        den = num1 + num2
        nextstate[1] = (num1 * g * p_t1 + num2 * b)/(den * R)	#Next pₜ
        nextstate[2] = p_t1 #Next pₜ_₁
        nextstate[3] = p_t2 #Next pₜ_2
    end
    parameters = a
    initial_state = [p₀, p₁, p₂] # Three dimensional dynamical system!
    return DiscreteDynamicalSystem(evolution!, initial_state, parameters)
end



T = 200
p₀ = 0.1
p₁ = 0.1
p₂ = 0.2
ds = generateinstance([1.3, 1.3, 1.0], p₀, p₁, p₂)
evolution = plot(trajectory(ds, T)[:, 1],marker = (:circle, :black),label = nothing, title = L"Evolution", xlabel = L"t", ylabel = L"P_t")
#savefig(evolution, "time_evolution.png")

function computeorbit(
    ds, alimits; 
    n = 1_000, 	# Number of points
    L = 500,
    xindex = 1, 	# Index of the state variables in the dynamical system
    aindex = 1, 	# Index of the parameter in the dynamical system
    kwargs...
)

a0, a1 = alimits
P = range(a0, a1, length = L)
orbits = orbitdiagram(
    ds, xindex, aindex, P; 
    n = n, Ttr = 2000, kwargs...
)
x = Vector{Float64}(undef, n*L) # Empty vector to store points
y = copy(x)
for j in 1:L
    x[(1 + (j-1)*n):j*n] .= P[j]
    y[(1 + (j-1)*n):j*n] .= orbits[j]
end
return x, y
end

x, y = computeorbit(ds, (0.0, 100.0), aindex = 1)
	
bifurcation = scatter(x, y,
	xaxis = L"beta", ylabel = L"p", 
	# ms = 1.5, color = :black, legend = nothing,
	# alpha = 0.1
)

#savefig(bifurcation, "two_types_bifurcation_g.png")
### Lyapunov Exponent

aspace = range(0.0, 0.4, length = 500)
λs = zeros(length(aspace))
	
for (i, a) in enumerate(aspace)
    set_parameter!(ds, 1, a)
    λs[i] = lyapunov(ds, 8_000; Ttr = 500)
end
	
plot(
	aspace, λs, 
	xlabel = "a", ylabel = "lambda", label = nothing,
	title = "Lyapunov exponent")

savefig(plot(
	aspace, λs, 
	xlabel = "a", ylabel = "lambda", label = nothing,
	title = "Lyapunov exponent"), "Lyapunov_exponent.png")


    using Flux
    function generateinstance_LSTM_speculator(a, p₂, p₁, p₀, LSTM_p₁, LSTM_p₀, hidden_state; R = 1.01)
        buffer = [initial_state] # Initialize buffer with the first state
        # Define the architecture of the LSTM network
        m = Chain(
            LSTM(1, 8), # input size 2, hidden size 8
            LSTM(8, 8), # hidden size 8, hidden size 8
            Dense(8, 1), # hidden size 8, output size 1
            tanh 
          )

        # Define a loss function and an optimizer
        loss(x, y) = Flux.mse(m(x), y)
        opt = ADAM()
        @inline @inbounds function evolution!(nextstate, state, parameters, t)
            p_t1, p_t2, p_t3, LSTM_pt2, LSTM_pt3  = state
            β, g, b = parameters
            prof1 = (p_t1 - R*p_t2)*(g*p_t3 - R*p_t2)
            prof2 = (p_t1 - R*p_t2)*(b -R*p_t2)
            prof3 = (p_t1 - R*p_t2)*(LSTM_pt3 - R*p_t2)
            
            #Train the LSTM model

            # Use previous states in buffer to make the LSTM forecast
            LSTM_pt1 = predict(LSTM, buffer)
            num1 = exp(β*(prof1))
            num2 = exp(β*(prof2))
            num3 = exp(β*(prof3))
            den = num1 + num2 + num3
            nextstate[1] = (num1 * g * p_t1 + num2 * b + num3 * LSTM_pt1)/(den * R)	#Next pₜ
            nextstate[2] = p_t1 #Next pₜ_₁
            nextstate[3] = p_t2 #Next pₜ_2
            nextstate[4] = LSTM_pt1
            nextstate[5] = LSTM_pt2
            nextstate[6] = #new hidden_state
            push!(buffer, nextstate) # Update the buffer with the new state
        end
        parameters = a
        initial_state = [p₂, p₁, p₀,LSTM_p₁, LSTM_p₀, hidden_state] 
        return DiscreteDynamicalSystem(evolution!, initial_state, parameters)
    end