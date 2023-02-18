## Environment setup (uncomment and run if needed)
# using Pkg
# Pkg.activate(temp=true) # Activate temporary environment
# Pkg.develop(path=dirname(@__DIR__)) # Develop GenGPT3 package  
# Pkg.add("Gen") # Add Gen package
# Pkg.add("PDDL") # Add PDDL package
# Pkg.add("PlanningDomains") # Add PlanningDomains package
# Pkg.add("SymbolicPlanners") # Add SymbolicPlanners package

# Load modules
using Gen, GenGPT3
using PDDL, PlanningDomains, SymbolicPlanners

# Register PDDL array theory
PDDL.Arrays.@register()

# Load domain and problem
domain = load_domain(JuliaPlannersRepo, "doors-keys-gems")
problem = load_problem(JuliaPlannersRepo, "doors-keys-gems", 3)

##  Define synchronous plan-to-narrative model

# Examples of annotated actions
annotated_actions = """
(right) ; They go right.
(unlock key1 door3) ; They use a key to unlock the door.
(left) ; They move left.
(up) ; They move upwards.
(pickup key1); They pick-up a key.
(up) ; They go up.
(left) ; They go leftwards.
(pickup gem2); They collect a gem.
(down) ; They move one step down.
(right) ; They move to the right.
(unlock key2 door1) ; They unlock a door with a key.
(down) ; They go downwards.
"""

@dist labeled_uniform(labels) = labels[uniform_discrete(1, length(labels))]

gpt3 = GPT3GF(max_tokens=32, stop="\n", model="text-babbage-001")

"Sample a random action, generate a description, then advance to next state."
@gen function sync_narrative_step(t::Int, step_info, domain::Domain)
    # Extract state
    state = step_info[3]
    # Sample an available action at random
    actions = available(domain, state)
    act ~ labeled_uniform(actions)
    # Generate natural language description given action
    prompt = annotated_actions * write_pddl(act) * " ;"
    desc ~ gpt3(prompt)
    # Advance to next state
    state = transition(domain, state, act; check=false)
    # Construct and return step information
    step_info = (act, desc, state)
    return step_info
end

sync_narrative_unfold = Unfold(sync_narrative_step)

"Samples a sequence of actions and a corresponding synchronous narrative."
@gen (static) function sync_narrative_model(T::Int, domain::Domain, state::State)
    init_step_info = (PDDL.no_op, "", state)
    steps = {:timestep} ~ sync_narrative_unfold(T, init_step_info, domain)
    plan = broadcast(getindex, steps, 1)
    narrative = broadcast(getindex, steps, 2)
    return plan, narrative
end

Gen.@load_generated_functions()

# Generate an example action sequence and narrative from initial state

state = initstate(domain, problem)
plan, narrative = sync_narrative_model(5, domain, state)

# Perform step-wise greedy inference of a likely plan

"Infer a likely plan from a synchronous narrative through greedy decoding."
function sync_narrative_to_plan(domain::Domain, state::State, 
                                narrative::Vector{<:AbstractString};
                                verbose=true)
    init_state = state
    argdiffs = (UnknownChange(), NoChange(), NoChange())
    # Generate initial trace
    trace, weight = generate(sync_narrative_model, (0, domain, state))
    if verbose 
        println("t\t$(rpad("plan (inferred)", 20))\tnarrative")
        println("="^60)
    end
    # Iterate over sentences in narrative
    for (t, sentence) in enumerate(narrative)
        # Construct observation choicemap
        obs = choicemap((:timestep => t => :desc => :output, sentence))
        # Find action that best explains observed sentence
        best_act, best_trace, best_weight = nothing, nothing, -Inf
        for act in available(domain, state)
            # Construct choicemap from observed sentence and action
            choices = merge(choicemap((:timestep => t => :act, act)), obs)
            # Update trace with new choices and compute incremental weight
            next_trace, up_weight, _, _ =
                Gen.update(trace, (t, domain, init_state), argdiffs, choices)
            # Retain trace with highest incremental weight
            if up_weight > best_weight
                best_act = act
                best_weight = up_weight
                best_trace = next_trace
            end
        end
        if verbose
            println("$t\t$(rpad(write_pddl(best_act), 20))\t$sentence")
        end
        # Advance to next state and trace
        state = transition(domain, state, best_act; check=false)
        trace = best_trace
        weight += best_weight
    end
    return trace, weight
end

narrative = [
    " First, they move right.",
    " Then they go right again.",
    " And then rightwards once more.",
    " Now they go up.",
    " And then up again.",
    " They turn left.",
    " And move leftwards again.",
    " And then left once more.",
    " Now they move up.",
    " They move upwards again.",
    " Then they go up again.",
    " And one step up.",
    " Then up once more.",
    " Finally they pick up the gem."
]

trace, weight = sync_narrative_to_plan(domain, state, narrative; verbose=true)

# Print inferred plan
plan, narrative = get_retval(trace)
for act in plan
    println(write_pddl(act))
end
