## GPT-3 API and utilities ##

const NO_EOT_BIAS = Dict(string(GPT_EOT_ID) => -100)

"Return API key stored in the OPENAI_API_KEY environment variable."
lookup_openai_api_key() = get(ENV, "OPENAI_API_KEY", "")

"Return organization ID stored in the OPENAI_ORGANIZATION environment variable."
lookup_openai_organization() = get(ENV, "OPENAI_ORGANIZATION", "")

"Call the OpenAI JSON API for GPT-3 and return the results."
function gpt3_api_call(
    prompt, n_completions::Int=1;
    endpoint::String = "https://api.openai.com/v1/completions",
    api_key::String = lookup_openai_api_key(),
    organization::String = lookup_openai_organization(),
    n_retries::Int = 10,
    model::String = "text-davinci-003",
    temperature::Real = 1.0,
    max_tokens::Int = 1024,
    logprobs::Union{Nothing,Int} = 0,
    echo::Bool = false,
    stop::Union{Nothing,String} = nothing,
    logit_bias::Union{Dict,Nothing} = nothing,
    verbose::Bool = false,
    options... # Other options
)
    if temperature > 2.0
        @warn "Temperature $temperature is too high, setting to 2.0"
        temperature = 2.0
    end
    # Construct HTTP request headers and body
    headers = ["Content-Type" => "application/json",
               "Authorization" => "Bearer $api_key",
               "OpenAI-Organization" => organization]
    body = Dict{String,Any}(
        "prompt" => prompt,
        "n" => n_completions,
        "model" => model,
        "temperature" => temperature,
        "max_tokens" => max_tokens,
        "logprobs" => logprobs,
        "echo" => echo,
        "stop" => stop
    )
    for (key, value) in options
        body[string(key)] = value
    end
    if !isnothing(logit_bias)
        body["logit_bias"] = logit_bias
    end
    body = JSON3.write(body)
    # Post request with exponential backoff
    if verbose println("Posting HTTP request...") end
    delays = ExponentialBackOff(n=n_retries, first_delay=0.5, max_delay=60.0,
                                factor=2.0, jitter=0.1)
    request = Base.retry(delays=delays,
                         check=(_, e) -> HTTP.RetryRequest.isrecoverable(e)) do
        HTTP.post(endpoint, headers, body, retry=false)
    end
    response = request()
    return JSON3.read(response.body)
end

"Make batched requests to the OpenAI API to reach prompt quota."
function gpt3_multi_prompt_api_call(
    prompts::AbstractVector{<:Union{AbstractString, AbstractVector{Int}}};
    batch_size::Int=min(length(prompts), 16),
    verbose::Bool=false,
    options...
)
    n_remaining = length(prompts)
    choices = JSON3.Object[]
    for batch in Iterators.partition(prompts, batch_size)
        n_request = length(batch)
        n_remaining -= n_request
        if verbose
            println("Making $n_request requests ($n_remaining remaining)...")
        end
        if all(prompt == batch[1] for prompt in batch)
            prompt = batch[1]
            new_choices = gpt3_multi_completion_api_call(
                prompt, n_request; batch_size=n_request,
                verbose=verbose, options...
            )
            append!(choices, new_choices)
        else
            response = gpt3_api_call(batch; verbose=verbose, options...)
            n_received = length(choices)
            resize!(choices, n_received + n_request)
            for choice in response.choices
                idx = n_received + choice.index + 1
                choices[idx] = choice
            end
        end
    end
    return choices
end

"Make batched requests to the OpenAI API to reach completion quota."
function gpt3_multi_completion_api_call(
    prompt::Union{AbstractString, AbstractVector{Int}}, n_completions::Int;
    batch_size::Int=min(n_completions, 16),
    verbose::Bool=false, options...
)
    if n_completions > 1 && get(options, :max_tokens, nothing) == 0
        response = gpt3_api_call(prompt, 1; verbose=verbose, options...)
        choices = fill(response.choices[1], n_completions)
        return choices
    end
    n_remaining = n_completions
    choices = JSON3.Object[]
    while n_remaining > 0
        n_request = min(n_remaining, batch_size)
        n_remaining -= n_request
        if verbose
            println("Making $n_request requests ($n_remaining remaining)...")
        end
        response = gpt3_api_call(prompt, n_request; verbose=verbose, options...)
        append!(choices, response.choices)
    end
    return choices
end

"Find the index of the completion's first token when a prompt is echoed."
function find_start_index(completion, prompt::String)
    text_offsets = completion.logprobs.text_offset
    if isempty(text_offsets)
        return 0
    else
        start_idx = findfirst(==(length(prompt)), text_offsets)
        return start_idx
    end
end

"Extract tokens and logprobs from completion after prompt."
function extract_tokens_after_prompt(completion, prompt::String)
    start_idx = find_start_index(completion, prompt)
    if start_idx > length(completion.logprobs.tokens) || start_idx == 0
        return String[], Float64[]
    else
        tokens = completion.logprobs.tokens[start_idx:end]
        logprobs = completion.logprobs.token_logprobs[start_idx:end]
        return tokens, logprobs
    end
    tokens = completion.logprobs.tokens[start_idx:end]
    logprobs = completion.logprobs.token_logprobs[start_idx:end]
    return tokens, collect(Float64, logprobs)
end

"""
Construct full text from prompt and completion output, appending stop sequence
and converting to token IDs if necessary. Returns nothing if the output (plus 
stop sequence) exceeds the maximum allowed number of tokens.
"""
function construct_full_text(
    max_tokens::Int, 
    prompt::String,
    output::String,
    stop::Union{String,Nothing},
    n_stop_tokens::Int = isnothing(stop) ? 1 : length(tokenize(stop))
)
    output_ids = id_tokenize(output)
    n_output_tokens = length(output_ids)
    if n_output_tokens + n_stop_tokens <= max_tokens # Finish due to stop words
        # Construct full text from prompt, output, and stop sequence
        if isnothing(stop)
            # Convert to token IDs, append <|endoftext|>
            prompt_ids = id_tokenize(prompt)
            full_text = append!(prompt_ids, output_ids, GPT_EOT_ID)
        else
            # Append stop sequence to text
            full_text = prompt * output * stop
        end
    elseif n_output_tokens == max_tokens # Finish due to length
        if isnothing(stop)
            # Convert to token IDs
            prompt_ids = id_tokenize(prompt)
            full_text = append!(prompt_ids, output_ids)
        else
            # Construct full text from prompt and output
            full_text = prompt * output
        end
    else # Full text could not have been generated
        full_text = nothing
    end
    return full_text
end

"Standardize logit bias input to a Dict{String,Float64}."
function standardize_logit_bias(logit_bias::Dict, stop=nothing)
    new_logit_bias = Dict{String,Float64}()
    if !isnothing(stop) && isempty(logit_bias)
        new_logit_bias[string(GPT_EOT_ID)] = -100
    end
    for (key, value) in logit_bias
        @assert value isa Real
        if key isa Int
            new_logit_bias[string(key)] = Float64(value)
        elseif !isnothing(tryparse(Int, key))
            new_logit_bias[string(parse(Int, key))] = Float64(value)
        elseif key isa String
            token_ids = id_tokenize(key)
            @assert length(token_ids) == 1 "$key is not a valid token"
            new_logit_bias[string(token_ids[1])] = Float64(value)
        else
            error("Invalid token identifief type: $(typeof(key))")
        end
    end
    return new_logit_bias
end

# If a stop token is specified, we ban <|endoftext|> from being generated
standardize_logit_bias(logit_bias::Nothing, stop) = NO_EOT_BIAS

standardize_logit_bias(logit_bias::Nothing, stop::Nothing) = logit_bias

"Call the OpenAI JSON API for text embeddings and return the results."
function embeddings_api_call(
    input;
    model::String = "text-embedding-ada-002",
    endpoint::String = "https://api.openai.com/v1/embeddings",
    api_key::String = lookup_openai_api_key(),
    organization::String = lookup_openai_organization(),
    n_retries::Int = 10,
    verbose::Bool = false,
    options...
)
    # Construct HTTP request headers and body
    headers = ["Content-Type" => "application/json",
               "Authorization" => "Bearer $api_key",
               "OpenAI-Organization" => organization]
    body = Dict{String,Any}(
        "input" => input,
        "model" => model
    )
    for (key, value) in options
        body[string(key)] = value
    end
    body = JSON3.write(body)
    # Post request with exponential backoff
    if verbose println("Posting HTTP request...") end
    delays = ExponentialBackOff(n=n_retries, first_delay=0.5, max_delay=60.0,
                                factor=2.0, jitter=0.1)
    request = Base.retry(delays=delays,
                         check=(_, e) -> HTTP.RetryRequest.isrecoverable(e)) do
        HTTP.post(endpoint, headers, body, retry=false)
    end
    response = request()
    return JSON3.read(response.body)
end
