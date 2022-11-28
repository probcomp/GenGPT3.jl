if GenGPT3.lookup_openai_api_key() == ""
    error("OPENAI_API_KEY not set, web API tests cannot run.")
end 

@testset "Web API" begin

    # Test API call with <|endoftext|> as stop sequence
    response = GenGPT3.gpt3_api_call(
        "What is the tallest mountain on Mars?", 3, model="text-babbage-001",
        logprobs=0, temperature=0.0, max_tokens=64
    )
    @test response.usage.prompt_tokens == 8
    @test length(response.choices) == 3
    
    # Check completion tokens are as expected
    completion = response.choices[1]
    output = completion.text
    stop_idx = GenGPT3.find_stop_index(completion, 1)
    tokens = completion.logprobs.tokens[1:stop_idx]
    logprobs = completion.logprobs.token_logprobs[1:stop_idx]
    @test tokens[end] == "<|endoftext|>"
    @test output == join(tokens[1:end-1])
    
    # Test API call with custom stop sequence
    response = GenGPT3.gpt3_api_call(
        "What is the tallest mountain on Mars?", 1, model="text-babbage-001",
        logprobs=0, temperature=0.0, max_tokens=64, stop=" Mount Sharp"
    )
    @test response.usage.prompt_tokens == 8
    @test length(response.choices) == 1
    
    # Check completion tokens are as expected
    completion = response.choices[1]
    output = completion.text
    n_stop_tokens = length(GenGPT3.tokenize(" Mount Sharp"))
    stop_idx = GenGPT3.find_stop_index(completion, n_stop_tokens)
    tokens = completion.logprobs.tokens[1:stop_idx]
    logprobs = completion.logprobs.token_logprobs[1:stop_idx]
    @test tokens[end-1:end] == [" Mount", " Sharp"]
    @test output == join(tokens[1:end-2])

end
