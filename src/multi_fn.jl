## MultiGPT3ChoiceMap ##

"""
    MultiGPT3ChoiceMap

An alias for the choicemap associated with [`MultiGPT3Trace`](@ref). 

    choices = MultiGPT3ChoiceMap(outputs::AbstractVector{String})

Constructs a choicemap for the trace of a [`MultiGPT3GenerativeFunction`](@ref).
"""
const MultiGPT3ChoiceMap = Gen.InternalVectorChoiceMap{GPT3ChoiceMap}

MultiGPT3ChoiceMap(outputs::AbstractVector{String}) =
    Gen.InternalVectorChoiceMap(GPT3ChoiceMap.(outputs), isempty(outputs))

## MultiGPT3Trace ##

"""
    MultiGPT3Trace

A trace generated by a [`MultiGPT3GenerativeFunction`](@ref). Effectively
represents a vector of [`GPT3Trace`](@ref) subtraces.

`Base.getindex(trace::MultiGPT3Trace, addr)` supports the
following values for `addr`:
- `:prompts`: The prompts provided to the language model.
- `:outputs`: The outputs generated for each prompt.
- `:tokens`: Vector of vectors of output tokens, including the stop sequence.
- `:token_logprobs`: Vector of vectors of token log probabilities.
- `:output_scores`: The log probabilities of each output.
- `:score`: The total log probability of generating all outputs.

In addition, `addr` can take the form `i => subaddr`, where `i` is an integer,
in order to access the `:prompt`, `:output`, `:tokens`, `:token_logprobs`,
or `:score` for the `i`-th subtrace.
"""
struct MultiGPT3Trace{T <: GenerativeFunction} <: Trace
    gen_fn::T
    prompts::Vector{String}
    outputs::Vector{String}
    tokens::Vector{Vector{String}}
    logprobs::Vector{Vector{Float64}}
    scores::Vector{Float64}
    score::Float64
end

function MultiGPT3Trace(gen_fn::GenerativeFunction)
    return MultiGPT3Trace(gen_fn, String[], String[], Vector{String}[], 
                          Vector{Float64}[], Float64[], 0.0)
end

get_choices(trace::MultiGPT3Trace) = MultiGPT3ChoiceMap(trace.outputs)
get_args(trace::MultiGPT3Trace) = (trace.prompts,)
get_retval(trace::MultiGPT3Trace) = trace.outputs
get_score(trace::MultiGPT3Trace) = trace.score
get_gen_fn(trace::MultiGPT3Trace) = trace.gen_fn

Base.getindex(trace::MultiGPT3Trace, idx::Int) = trace.outputs[i]

function Base.getindex(trace::MultiGPT3Trace, addr::Symbol)
    if addr == :prompts
        return trace.prompts
    elseif addr == :tokens
        return trace.tokens
    elseif addr == :token_logprobs
        return trace.logprobs
    elseif addr == :outputs
        return trace.outputs
    elseif addr == :output_scores
        return trace.scores
    elseif addr == :score
        return trace.score
    else
        throw(KeyError(addr))
    end
end

function Base.getindex(trace::MultiGPT3Trace, addr::Pair{Int, Symbol})
    i, addr = addr
    if addr == :prompt
        return trace.prompts[i]
    elseif addr == :tokens
        return trace.tokens[i]
    elseif addr == :token_logprobs
        return trace.logprobs[i]
    elseif addr == :output
        return trace.outputs[i]
    elseif addr == :score
        return trace.scores[i]
    else
        throw(KeyError(addr))
    end
end

function Base.:(==)(trace1::MultiGPT3Trace, trace2::MultiGPT3Trace)
    return (trace1.gen_fn == trace2.gen_fn &&
            trace1.prompts == trace2.prompts &&
            trace1.outputs == trace2.outputs &&
            trace1.tokens == trace2.tokens &&
            trace1.logprobs == trace2.logprobs &&
            trace1.scores == trace2.scores &&
            trace1.score == trace2.score)
end

function Base.vcat(trace1::MultiGPT3Trace, trace2::MultiGPT3Trace)
    return MultiGPT3Trace(
        trace1.gen_fn,
        vcat(trace1.prompts, trace2.prompts),
        vcat(trace1.outputs, trace2.outputs),
        vcat(trace1.tokens, trace2.tokens),
        vcat(trace1.logprobs, trace2.logprobs),
        vcat(trace1.scores, trace2.scores),
        trace1.score + trace2.score
    )
end

## MultiGPT3GenerativeFunction ##

"""
    MultiGPT3GenerativeFunction(;
        model = "davinci-002",
        temperature = 1.0,
        max_tokens = 1024,
        stop = nothing,
        batch_size = 10,
        encoding = GenGPT3.MODEL_ENCODINGS[model],
        api_key_lookup = () -> ENV["OPENAI_API_KEY"],
        organization_lookup = () -> ENV["OPENAI_ORGANIZATION"]
    )

Batched version of [`GPT3GenerativeFunction`](@ref), which requests and 
returns completions for a batch of prompts. Takes in a `Vector` of `String`
valued prompts as an argument. The completion for the `i`th prompt is stored
in the `i => $OUTPUT_ADDR` address of the resulting trace.
"""
@kwdef struct MultiGPT3GenerativeFunction <: GenerativeFunction{String,MultiGPT3Trace}
    model::String = "davinci-002"
    temperature::Float64 = 1.0
    max_tokens::Int = 1024
    encoding::String = MODEL_ENCODINGS[model]
    stop::Union{String,Nothing} = nothing
    n_stop::Int = isnothing(stop) ? 1 : length(tokenize(encoding, stop))
    batch_size::Int = DEFAULT_BATCH_SIZE
    api_key_lookup::Function = lookup_openai_api_key
    organization_lookup::Function = lookup_openai_organization
end

"""
    MultiGPT3GF

An alias for [`MultiGPT3GenerativeFunction`](@ref).
"""
const MultiGPT3GF = MultiGPT3GenerativeFunction

"""
    (gpt3::MultiGPT3GenerativeFunction)(prompts::Vector{String})

Untraced execution of a [`MultiGPT3GenerativeFunction`]. Calls GPT-3 with
a batch of prompts, and returns the resulting completions.
"""
function (gen_fn::MultiGPT3GenerativeFunction)(prompts::Vector{String})
    n = length(prompts)
    outputs = Vector{String}(undef, n)
    # Request completions through GPT-3 API
    choices = gpt3_multi_prompt_api_call(
        prompts;
        batch_size=min(gen_fn.batch_size, n),
        model=gen_fn.model,
        temperature=gen_fn.temperature,
        max_tokens=gen_fn.max_tokens,
        stop=gen_fn.stop,
        logit_bias=standardize_logit_bias(nothing, gen_fn.stop, gen_fn.encoding),
        api_key=gen_fn.api_key_lookup(),
        organization=gen_fn.organization_lookup()
    )
    # Extract outputs
    for (i, completion) in enumerate(choices)
        outputs[i] = completion.text
    end
    return outputs
end

(gen_fn::MultiGPT3GenerativeFunction)(n::Int, prompt::String) =
    gen_fn(fill(prompt, n))
    
function simulate(gen_fn::MultiGPT3GF, args::Tuple{Vector{String}})
    # Extract prompts and initialize arrays
    prompts = args[1]
    n = length(prompts)

    # Decide whether to sample and score in the same API call
    if all(p === prompts[1] for p in prompts)
        n_prompt_tokens = length(tokenize(gen_fn.encoding, prompts[1]))
        same_call = n_prompt_tokens > gen_fn.max_tokens && !isnothing(gen_fn.stop)
    else
        same_call = false
    end
    stop = same_call ? nothing : gen_fn.stop
    logit_bias = same_call ?
        NO_EOT_BIAS : standardize_logit_bias(nothing, gen_fn.stop, gen_fn.encoding)

    # Request completions through GPT-3 API
    choices = gpt3_multi_prompt_api_call(
        prompts;
        batch_size=min(gen_fn.batch_size, n),
        model=gen_fn.model,
        temperature=gen_fn.temperature,
        max_tokens=gen_fn.max_tokens,
        stop=stop,
        logit_bias=logit_bias,
        api_key=gen_fn.api_key_lookup(),
        organization=gen_fn.organization_lookup()
    )

    # Score completions by calling `generate` if necessary
    if !same_call
        outputs = Vector{String}(undef, n)
        for (i, completion) in enumerate(choices)
            outputs[i] = completion.text
        end
        trace, _ = generate(gen_fn, args, MultiGPT3ChoiceMap(outputs))
        return trace
    end

    # Construct trace from completions
    outputs = Vector{String}(undef, n)
    tokens = Vector{Vector{String}}(undef, n)
    logprobs = Vector{Vector{Float64}}(undef, n)
    scores = Vector{Float64}(undef, n)
    for (i, completion) in enumerate(choices)
        tokens[i], lps = extract_tokens_until_stop(completion, gen_fn.stop;
                                                   encoding=gen_fn.encoding)
        logprobs[i] = gen_fn.temperature == 0.0 ?
            zeros(Float64, length(tokens)) : lps ./ gen_fn.temperature
        scores[i] = isempty(logprobs[i]) ? 0.0 : sum(logprobs[i])
        outputs[i] = join(tokens[i][1:end-gen_fn.n_stop])
    end
    total_score = sum(scores)
    trace = MultiGPT3Trace(gen_fn, prompts, outputs, tokens,
                           logprobs, scores, total_score)
    return trace
end

simulate(gen_fn::MultiGPT3GF, args::Tuple{Int, String}) =
    simulate(gen_fn, (fill(args[2], args[1]),))

function generate(gen_fn::MultiGPT3GF, args::Tuple, constraints::ChoiceMap)
    # Check whether any outputs are constrained
    if isempty(constraints)
        return generate(gen_fn, args, EmptyChoiceMap())
    end

    # Extract prompts and initialize arrays
    prompts = args[1]
    n = length(prompts)
    outputs = Vector{String}(undef, n)
    tokens = Vector{Vector{String}}(undef, n)
    logprobs = Vector{Vector{Float64}}(undef, n)
    scores = Vector{Float64}(undef, n)

    # Extract constrained outputs and construct full texts
    constrained_idxs = Int[]
    unconstrained_idxs = Int[]
    impossible = false
    full_texts = Vector{Int}[]
    for i in eachindex(outputs)
        addr = i => OUTPUT_ADDR
        if !has_value(constraints, addr)
            push!(unconstrained_idxs, i)
            continue
        end
        outputs[i] = constraints[addr]
        full_text = construct_full_text(gen_fn.max_tokens,
                                        prompts[i], outputs[i],
                                        gen_fn.stop, gen_fn.n_stop;
                                        encoding=gen_fn.encoding)
        # If nothing is returned, then the constrained output is too long
        if isnothing(full_text)
            scores[i] = -Inf
            impossible = true
            continue
        end
        push!(full_texts, full_text)
        push!(constrained_idxs, i) 
    end

    # Score the full texts for constrained indices
    if !isempty(constrained_idxs)
        choices = gpt3_multi_prompt_api_call(
            full_texts;
            batch_size=min(gen_fn.batch_size, length(full_texts)),
            logprobs=0,
            model=gen_fn.model,
            temperature=gen_fn.temperature,
            max_tokens=0,
            echo=true,
            stop=gen_fn.stop,
            logit_bias=standardize_logit_bias(nothing, gen_fn.stop, gen_fn.encoding),
            api_key=gen_fn.api_key_lookup(),
            organization=gen_fn.organization_lookup()
        )

        # Extract scores from returned completions
        for (i, completion) in zip(constrained_idxs, choices)
            tokens[i], token_lps =
                extract_tokens_after_prompt(completion, prompts[i])
            logprobs[i] = gen_fn.temperature == 0.0 ?
                zeros(Float64, length(token_lps)) : token_lps ./ gen_fn.temperature
            scores[i] = isempty(logprobs[i]) ? 0.0 : sum(logprobs[i])
        end
    end

    # Sample all unconstrained choices
    if !isempty(unconstrained_idxs)
        partial_trace = simulate(gen_fn, (prompts[unconstrained_idxs],))
        for (k, i) in enumerate(unconstrained_idxs)
            outputs[i] = partial_trace.outputs[k]
            tokens[i] = partial_trace.tokens[k]
            logprobs[i] = partial_trace.logprobs[k]
            scores[i] = partial_trace.scores[k]
        end
    end

    # Construct and return trace and weight
    total_score = isempty(scores) ? 0.0 : sum(scores)
    weight = if impossible
        -Inf
    elseif isempty(constrained_idxs)
        0.0
    else
        sum(scores[constrained_idxs])
    end
    trace = MultiGPT3Trace(gen_fn, prompts, outputs, tokens,
                           logprobs, scores, total_score)
    return trace, weight
end

generate(gen_fn::MultiGPT3GF, args::Tuple{Int, String}, constraints::ChoiceMap) =
    generate(gen_fn, (fill(args[2], args[1]),), constraints)

generate(gen_fn::MultiGPT3GF, args::Tuple, ::EmptyChoiceMap) =
    simulate(gen_fn, args), 0.0

generate(gen_fn::MultiGPT3GF, args::Tuple{Int, String}, ::EmptyChoiceMap) =
    simulate(gen_fn, args), 0.0

function project(trace::MultiGPT3Trace, selection::Selection)
    if isempty(selection) return 0.0 end
    weight = 0.0
    for i in eachindex(trace.prompts)
        addr = i => OUTPUT_ADDR
        weight += addr in selection ? trace.scores[i] : 0.0
    end
    return weight
end

project(trace::MultiGPT3Trace, ::AllSelection) = trace.score

project(trace::MultiGPT3Trace, ::EmptySelection) = 0.0

function update(trace::MultiGPT3Trace, args::Tuple,
                argdiffs::Tuple, constraints::ChoiceMap)
    gen_fn = trace.gen_fn

    # Extract prompts and copy arrays from old trace
    old_prompts, new_prompts = trace.prompts, args[1]
    old_n, new_n = length(old_prompts), length(new_prompts)
    outputs = resize!(copy(trace.outputs), new_n)
    tokens = resize!(copy(trace.tokens), new_n)
    logprobs = resize!(copy(trace.logprobs), new_n)
    scores = resize!(copy(trace.scores), new_n)

    # Extract updated outputs and construct full texts
    updated_idxs = Int[]
    created_idxs = Int[]
    impossible = false
    full_texts = Vector{Int}[]
    for i in 1:new_n
        addr = i => OUTPUT_ADDR
        if !has_value(constraints, addr)
            if i > old_n
                push!(created_idxs, i)
                continue
            elseif new_prompts[i] == old_prompts[i]
                continue
            end
        else
            outputs[i] = constraints[addr] # Output was updated
        end
        full_text = construct_full_text(gen_fn.max_tokens,
                                        new_prompts[i], outputs[i],
                                        gen_fn.stop, gen_fn.n_stop;
                                        encoding=gen_fn.encoding)
        # If nothing is returned, then the constrained output is too long
        if isnothing(full_text)
            scores[i] = -Inf
            impossible = true
            continue
        end
        push!(full_texts, full_text)
        push!(updated_idxs, i) 
    end

    # Score the full texts for updated indices
    if !isempty(updated_idxs)
        choices = gpt3_multi_prompt_api_call(
            full_texts;
            batch_size=min(gen_fn.batch_size, length(full_texts)),
            logprobs=0,
            model=gen_fn.model,
            temperature=gen_fn.temperature,
            max_tokens=0,
            echo=true,
            stop=gen_fn.stop,
            logit_bias=standardize_logit_bias(nothing, gen_fn.stop, gen_fn.encoding),
            api_key=gen_fn.api_key_lookup(),
            organization=gen_fn.organization_lookup()
        )

        # Extract scores from returned completions
        for (i, completion) in zip(updated_idxs, choices)
            tokens[i], token_lps =
                extract_tokens_after_prompt(completion, new_prompts[i])
            logprobs[i] = gen_fn.temperature == 0.0 ?
                zeros(Float64, length(token_lps)) : token_lps ./ gen_fn.temperature
            scores[i] = isempty(logprobs[i]) ? 0.0 : sum(logprobs[i])
        end
    end

    # Sample completions for newly created indices
    if !isempty(created_idxs)
        partial_trace = simulate(gen_fn, (new_prompts[created_idxs],))
        for (k, i) in enumerate(created_idxs)
            outputs[i] = partial_trace.outputs[k]
            tokens[i] = partial_trace.tokens[k]
            logprobs[i] = partial_trace.logprobs[k]
            scores[i] = partial_trace.scores[k]
        end
    end

    # Compute total score and construct new trace
    total_score = isempty(scores) ? 0.0 : sum(scores)
    new_trace = MultiGPT3Trace(gen_fn, new_prompts, outputs, tokens,
                               logprobs, scores, total_score)

    # Compute incremental weight and discarded choices
    weight = 0.0
    discard = choicemap()
    for i in updated_idxs # Contributions from updated indices
        weight += new_trace.scores[i]
        if i <= old_n
            weight -= trace.scores[i]
        end
        if has_value(constraints, i => OUTPUT_ADDR)
            discard[i => OUTPUT_ADDR] = trace.outputs[i]
        end
    end
    for i in (new_n+1):old_n # Contributions from discarded indices
        weight -= trace.scores[i]
        discard[i => OUTPUT_ADDR] = trace.outputs[i]
    end
    weight += impossible ? -Inf : 0

    # Compute diff and return
    retdiff = new_n == old_n && isempty(constraints) && !impossible ?
        NoChange() : UnknownChange()
    return new_trace, weight, retdiff, discard
end

function regenerate(trace::MultiGPT3Trace, args::Tuple,
                    argdiffs::Tuple, selection::Selection)
    gen_fn = trace.gen_fn

    # Extract prompts and copy arrays from old trace
    old_prompts, new_prompts = trace.prompts, args[1]
    old_n, new_n = length(old_prompts), length(new_prompts)
    outputs = resize!(copy(trace.outputs), new_n)
    tokens = resize!(copy(trace.tokens), new_n)
    logprobs = resize!(copy(trace.logprobs), new_n)
    scores = resize!(copy(trace.scores), new_n)

    # Extract selected or updated outputs and construct full texts
    regenerated_idxs = Int[]
    updated_idxs = Int[]
    full_texts = Vector{Int}[]
    for i in 1:new_n
        addr = i => OUTPUT_ADDR
        if i > old_n || addr in selection
            push!(regenerated_idxs, i)
            continue
        elseif new_prompts[i] == old_prompts[i]
            continue
        end
        full_text = construct_full_text(gen_fn.max_tokens,
                                        new_prompts[i], outputs[i],
                                        gen_fn.stop, gen_fn.n_stop;
                                        encoding=gen_fn.encoding)
        push!(full_texts, full_text)
        push!(updated_idxs, i) 
    end

    # Score the full texts for updated indices
    if !isempty(updated_idxs)
        choices = gpt3_multi_prompt_api_call(
            full_texts;
            batch_size=min(gen_fn.batch_size, length(full_texts)),
            logprobs=0,
            model=gen_fn.model,
            temperature=gen_fn.temperature,
            max_tokens=0,
            echo=true,
            stop=gen_fn.stop,
            logit_bias=standardize_logit_bias(nothing, gen_fn.stop, gen_fn.encoding),
            api_key=gen_fn.api_key_lookup(),
            organization=gen_fn.organization_lookup()
        )

        # Extract scores from returned completions
        for (i, completion) in zip(updated_idxs, choices)
            tokens[i], token_lps =
                extract_tokens_after_prompt(completion, new_prompts[i])
            logprobs[i] = gen_fn.temperature == 0.0 ?
                zeros(Float64, length(token_lps)) : token_lps ./ gen_fn.temperature
            scores[i] = isempty(logprobs[i]) ? 0.0 : sum(logprobs[i])
        end
    end

    # Sample completions for selected and newly created indices
    if !isempty(regenerated_idxs)
        partial_trace = simulate(gen_fn, (new_prompts[regenerated_idxs],))
        for (k, i) in enumerate(regenerated_idxs)
            outputs[i] = partial_trace.outputs[k]
            tokens[i] = partial_trace.tokens[k]
            logprobs[i] = partial_trace.logprobs[k]
            scores[i] = partial_trace.scores[k]
        end
    end

    # Compute total score and construct new trace
    total_score = isempty(scores) ? 0.0 : sum(scores)
    new_trace = MultiGPT3Trace(gen_fn, new_prompts, outputs, tokens,
                               logprobs, scores, total_score)

    # Compute incremental weight 
    weight = 0.0
    for i in updated_idxs # Contributions from updated indices
        weight += new_trace.scores[i] - trace.scores[i]
    end

    # Compute diff and return
    retdiff = new_n == old_n && isempty(regenerated_idxs) ?
        NoChange() : UnknownChange()
    return new_trace, weight, retdiff
end
