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

# Generate an example action sequence and narrative from the initial state

state = initstate(domain, problem)
plan, narrative = sync_narrative_model(5, domain, state)

##  Perform step-wise greedy inference of a likely plan

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

trace, weight = sync_narrative_to_plan(domain, state, narrative, verbose=true)

# Print inferred plan
plan, narrative = get_retval(trace)
for act in plan
    println(write_pddl(act))
end

##  Define *asynchronous* plan-to-narrative model

annotated_plan = """
;; Below is an annotated PDDL plan.
(right) (right) (right) ; First, they move three steps to the right.
(up) (up) ; Then up twice.
(left) (left) (left) ; Now they go all the way left.
(up) (up) (up) (up) ; Next, they move up several times.
(pickup key1) ; They pick up a key.
(right) (right) (unlock key1 door1) ; Then they go right, and use the key to unlock a door.
(right) (right) ; They go right through the corridoor.
(up) (right) (right) (right) ; Then turn up once, and go right three times.
(pickup gem2) ; Finally they reach and collect the gem.
"""

gpt3_curie = GPT3GF(max_tokens=32, stop="\n", model="text-curie-001")
gpt3_davinci = GPT3GF(max_tokens=32, stop="\n", model="text-davinci-002")
gpt3_codex = GPT3GF(max_tokens=32, stop="\n", model="code-davinci-002")

"Sample an action and decide whether to generate a description of recent actions."
@gen function async_narrative_step(t::Int, step_info, domain::Domain, lm)
    # Extract previous state and prompt
    state, prompt = step_info[3], step_info[4]
    # Sample an available action at random
    actions = available(domain, state)
    act ~ labeled_uniform(actions)
    # Append newest action to prompt
    prompt = prompt * write_pddl(act) * " "
    # Decide whether a description is generated at this step
    has_desc ~ bernoulli(1/3)
    if has_desc
        # Generate natural language description given prompt so far
        prompt = prompt * ";"
        desc ~ lm(prompt)
        prompt = prompt * desc * "\n"
    else
        desc = missing
    end
    # Advance to next state
    state = transition(domain, state, act; check=false)
    # Construct and return step information
    step_info = (act, desc, state, prompt)
    return step_info
end

async_narrative_unfold = Unfold(async_narrative_step)

"Samples a plan and a corresponding asynchronous narrative."
@gen (static) function async_narrative_model(
    T::Int, domain::Domain, state::State, lm::GenerativeFunction
)
    prompt = annotated_plan * "\n;; Below is another annotated PDDL plan.\n"
    init_step_info = (PDDL.no_op, "", state, prompt)
    steps = {:timestep} ~ async_narrative_unfold(T, init_step_info, domain, lm)
    plan = broadcast(getindex, steps, 1)
    narrative = broadcast(getindex, steps, 2)
    prompt = T > 0 ? steps[end][end] : ""
    return plan, narrative, prompt
end

Gen.@load_generated_functions()

# Generate an example action sequence and narrative from the initial state

state = initstate(domain, problem)
state[pddl"(xpos)"] = 1
state[pddl"(ypos)"] = 6
plan, narrative, prompt = async_narrative_model(5, domain, state, gpt3_davinci)

## Perform segment-wise greedy inference of a likely plan

"Enumerates all possible action sequences of length `n` from an initial state."
function enumerate_act_sequences(domain::Domain, init_state::State, n::Int)
    seq_iter = (([act], transition(domain, init_state, act; check=false))
                for act in available(domain, init_state))
    for i in 2:n
        seq_iter = Iterators.map(seq_iter) do (seq, state)
            return (([seq; act], transition(domain, state, act; check=false))
                    for act in available(domain, state))
        end
        seq_iter = Iterators.flatten(seq_iter)
    end
    return seq_iter
end

"Convert a sequence of actions from `t_start` to `t_stop` into a choicemap."
function act_choicemap(actions, t_start::Int, t_stop::Int)
    timesteps = t_start:t_stop
    @assert length(actions) == length(timesteps)
    choices = choicemap()
    for (t, act) in zip(timesteps, actions)
        choices[:timestep => t => :act] = act
    end
    return choices
end

"Convert a description observed at `t_stop` into a segment choicemap."
function desc_choicemap(desc::AbstractString, t_start::Int, t_stop::Int)
    choices = choicemap()
    for t in t_start:t_stop-1
        choices[:timestep => t => :has_desc] = false
    end
    choices[:timestep => t_stop => :has_desc] = true
    choices[:timestep => t_stop => :desc => :output] = desc
    return choices
end

"Infer a likely plan from an asynchronous narrative."
function async_narrative_to_plan(
    domain::Domain, state::State, lm::GenerativeFunction,
    timesteps::Vector{Int}, narrative::Vector{<:AbstractString};
    verbose=true
)
    init_state = state
    argdiffs = (UnknownChange(), NoChange(), NoChange(), NoChange())
    # Generate initial trace
    trace, weight = generate(async_narrative_model, (0, domain, state, lm))
    if verbose 
        println("t\t$(rpad("plan (inferred)", 30))\tnarrative")
        println("="^60)
    end
    # Iterate over sentences in narrative
    prev_t = 0
    for (t, sentence) in zip(timesteps, narrative)
        dur = t - prev_t
        # Construct observation choicemap
        obs_choices = desc_choicemap(sentence, prev_t+1, t)
        # Find action sequence that best explains observed sentence
        best_seq, best_state = nothing, nothing
        best_trace, best_weight = nothing, -Inf
        for (act_seq, next_state) in enumerate_act_sequences(domain, state, dur)
            # Construct choicemap from action sequence and observed sentence
            act_choices = act_choicemap(act_seq, prev_t+1, t)
            choices = merge(act_choices, obs_choices)
            # Update trace with new choices and compute incremental weight
            next_trace, up_weight, _, _ =
                Gen.update(trace, (t, domain, init_state, lm),
                           argdiffs, choices)
            # Retain trace with highest incremental weight
            if up_weight > best_weight
                best_seq = act_seq
                best_state = next_state
                best_weight = up_weight
                best_trace = next_trace
            end
        end
        if verbose
            act_seq_str = join(write_pddl.(best_seq), " ")
            println("$t\t$(rpad(act_seq_str, 30))\t$sentence")
        end
        # Advance to next state and trace
        state = best_state
        trace = best_trace
        weight += best_weight
        # Update timestep
        prev_t = t
    end
    return trace, weight
end

# Construct time-stamped asynchronous test narrative
timesteps = [4, 5, 9, 12, 14, 15, 17, 21, 23, 24]
narrative = [
    "First, they move up four times.",
    "Then, they pick up the key.",
    "Now they go back down to where they started.",
    "And then move right 3 steps.",
    "They go upwards twice.",
    "Then use the key to unlock the door.",
    "Then move right two steps.",
    "Next they go down all the way.",
    "Then they move right for a few steps.",
    "At last, they pick-up the gem."
]

# Run inference
trace, weight = async_narrative_to_plan(
    domain, state, gpt3_davinci,
    timesteps, narrative, verbose=true
)

# Print inferred plan
plan, narrative = get_retval(trace)
for act in plan
    println(write_pddl(act))
end