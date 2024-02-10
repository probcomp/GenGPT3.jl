if GenGPT3.lookup_openai_api_key() == ""
    error("OPENAI_API_KEY not set, web API tests cannot run.")
end 

@testset "Web API" begin

    # Test API call with <|endoftext|> as stop sequence
    response = GenGPT3.gpt3_api_call(
        "What is the tallest mountain on Mars?", 3, model="davinci-002",
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
        "What is the tallest mountain on Mars?", 1, model="davinci-002",
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

    # Test token extraction after prompt
    prompt = "What is the tallest mountain on Mars?"
    output = "\n\nThe tallest mountain on Mars is Mount Sharp, " *
             "which is located in the Martian equator."
    all_tokens = vcat(GenGPT3.id_tokenize("cl100k_base", prompt),
                      GenGPT3.id_tokenize("cl100k_base", output))
    response = GenGPT3.gpt3_api_call(
        all_tokens, 1, model="davinci-002", echo=true,
        logprobs=0, temperature=0.0, max_tokens=0
    )

    completion = response.choices[1]
    start_idx = GenGPT3.find_start_index(completion, prompt)
    @test start_idx == length(GenGPT3.tokenize("cl100k_base", prompt)) + 1
    output_tokens, _ = GenGPT3.extract_tokens_after_prompt(completion, prompt)
    @test join(output_tokens) == output

    # Test token extraction up to stop token
    prompt = "2 + 2 ="
    output = " 4."
    stop = "."
    response = GenGPT3.gpt3_api_call(
        prompt, 1, model="davinci-002",
        logit_bias=GenGPT3.standardize_logit_bias(nothing, "."),
        logprobs=0, temperature=0.0, max_tokens=32
    )

    completion = response.choices[1]
    stop_idx = GenGPT3.find_stop_index(completion, stop)
    @test stop_idx == length(GenGPT3.tokenize("cl100k_base", output))
    output_tokens, _ = GenGPT3.extract_tokens_until_stop(completion, stop)
    @test join(output_tokens) == output

    # Test multi-prompt API call
    prompts = [
        "Caloris Montes is the tallest mountain on Mercury.",
        "Skadi Mons is the tallest mountain on Venus.",
        "Mount Everest is the tallest mountain on Earth.",
        "Olympus Mons is the tallest mountain on Mars."
    ]
    choices = GenGPT3.gpt3_multi_prompt_api_call(
        prompts, batch_size=2, model="davinci-002",
        logprobs=0, max_tokens=0, echo=true
    )
    @test length(choices) == 4
    @test all(p == c.text for (p, c) in zip(prompts, choices))

    # Test multi-completion API call
    choices = GenGPT3.gpt3_multi_completion_api_call(
        "What is the tallest mountain on Mars?", 10,
        batch_size=4, model="davinci-002",
        logprobs=0, temperature=0.0, max_tokens=64
    )
    @test length(choices) == 10
    @test all(c.text == choices[1].text for c in choices)

    # Test embeddings API call
    response = GenGPT3.embeddings_api_call(
        ["What is the tallest mountain on Mars?",
         "What is the tallest mountain on Earth?"],
         model = "text-embedding-ada-002"
    )
    @test length(response.data) == 2
    @test length(response.data[1].embedding) == 1536 
    @test length(response.data[2].embedding) == 1536
end
