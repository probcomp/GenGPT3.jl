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
    tokens = completion.logprobs.tokens[1:end]
    @test output == join(tokens)
    
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
    tokens = completion.logprobs.tokens[1:end]
    @test tokens[end-1:end] != [" Mount", " Sharp"]
    @test output == join(tokens)

    # Test multi-prompt API call
    prompts = [
        "Caloris Montes is the tallest mountain on Mercury.",
        "Skadi Mons is the tallest mountain on Venus.",
        "Mount Everest is the tallest mountain on Earth.",
        "Olympus Mons is the tallest mountain on Mars."
    ]
    choices = GenGPT3.gpt3_multi_prompt_api_call(
        prompts, batch_size=2, model="text-babbage-001",
        logprobs=0, max_tokens=0, echo=true
    )
    @test length(choices) == 4
    @test all(p == c.text for (p, c) in zip(prompts, choices))

    # Test multi-completion API call
    choices = GenGPT3.gpt3_multi_completion_api_call(
        "What is the tallest mountain on Mars?", 10,
        batch_size=4, model="text-babbage-001",
        logprobs=0, temperature=0.0, max_tokens=64
    )
    @test length(choices) == 10
    @test all(c.text == choices[1].text for c in choices)

end
