## GPT-3 API and utilities ##

"Call the OpenAI JSON API for GPT-3 and return the results."
function gpt3_api_call(
    prompt, n_completions::Int=1;
    endpoint::String = "https://api.openai.com/v1/completions",
    api_key::String = get(ENV, "OPENAI_API_KEY", ""),
    model::String = "text-davinci-002",
    temperature::Real = 1.0,
    max_tokens::Int = 1024,
    logprobs::Union{Nothing,Int} = 0,
    echo::Bool = false,
    stop::Union{Nothing,String} = nothing,
    verbose::Bool = false,
    logit_bias::Union{Dict,Nothing} = nothing,
    options... # Other options
)
    headers = ["Content-Type" => "application/json",
               "Authorization" => "Bearer $api_key"]
    data = Dict{String,Any}(
        "prompt" => prompt,
        "n" => n_completions,
        "model" => model,
        "temperature" => temperature,
        "max_tokens" => max_tokens,
        "logprobs" => logprobs,
        "echo" => echo,
        "stop" => stop,
        options...
    )
    if !isnothing(logit_bias)
        data["logit_bias"] = logit_bias
    end
    data = JSON3.write(data)
    if verbose println("Posting HTTP request...") end
    response = HTTP.post(endpoint, headers, data)
    return JSON3.read(response.body)
end

"Return API key stored in the OPENAI_API_KEY environment variable."
lookup_openai_api_key() = get(ENV, "OPENAI_API_KEY", "")

"Find the index of the completion's first token when a prompt is echoed."
function find_start_index(completion, prompt::String)
    text_offsets = completion.logprobs.text_offset  
    start_idx = findfirst(==(length(prompt)), text_offsets)
    return start_idx
end

"Find the index of a completion's last token, including the stop sequence."
function find_stop_index(completion, n_stop_tokens::Int=1)
    if completion.finish_reason == "length"
        return length(completion.logprobs.tokens)
    elseif completion.finish_reason == "stop"
        text_offsets = completion.logprobs.text_offset  
        last_offset = text_offsets[end]
        first_stop_idx = findfirst(==(last_offset), text_offsets)
        last_stop_idx = first_stop_idx + n_stop_tokens - 1
        return last_stop_idx
    end
end
