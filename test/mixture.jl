if GenGPT3.lookup_openai_api_key() == ""
    error("OPENAI_API_KEY not set, GFI tests cannot run.")
end 

@testset "GPT3Mixture" begin

    gpt3_mixture = GPT3Mixture(model="davinci-002", max_tokens=64, stop="\n")

    @testset "trace" begin
        prompts = ["What is the tallest mountain on Mars?",
                   "Which mountain do the Greek gods live upon?"]
        probs = [0.25, 0.75]
        output = "\n\nMount Olympus."
        multi_trace, _ = generate(gpt3_mixture.multi_gf, (prompts,),
                                  MultiGPT3ChoiceMap(fill(output, 2)))
        tokens = multi_trace.tokens[1]
        scores = multi_trace.scores
        score = logsumexp(scores .+ log.(probs))
        trace = GPT3MixtureTrace(gpt3_mixture, prompts, probs, multi_trace,
                                 output, tokens, scores, score)

        @test get_choices(trace) == GPT3ChoiceMap(output)
        @test get_args(trace) == (prompts, probs)
        @test get_retval(trace) == output
        @test get_score(trace) == score
        @test get_gen_fn(trace) == gpt3_mixture

        @test trace[:prompts] == prompts
        @test trace[:prior_probs] == probs
        @test trace[:post_probs] == exp.(scores .+ log.(probs) .- score)
        @test trace[:joint_probs] == exp.(scores .+ log.(probs))
        @test trace[:tokens] == tokens
        @test trace[:output] == output
        @test trace[:output_scores] == scores
        @test trace[:score] == score
    end

    @testset "simulate" begin
        prompts = ["What is the tallest mountain on Mars?",
                   "Which mountain do the Greek gods live upon?"]
        probs = [0.25, 0.75]
        trace = simulate(gpt3_mixture, (prompts, probs))

        @test trace.prompts == prompts
        @test trace.probs == probs
        @test trace.output == (trace.tokens[end] == "\n" ? 
            join(trace.tokens[1:end-1]) : join(trace.tokens))
        @test trace.score == logsumexp(trace.scores .+ log.(probs))
        @test trace.gen_fn == gpt3_mixture
    end

    @testset "generate" begin
        # Unconstrained generation
        prompts = ["What is the tallest mountain on Mars?",
                   "Which mountain do the Greek gods live upon?"]
        probs = [0.25, 0.75]
        trace, weight = generate(gpt3_mixture, (prompts, probs))

        @test trace.prompts == prompts
        @test trace.probs == probs
        @test trace.output == (trace.tokens[end] == "\n" ? 
            join(trace.tokens[1:end-1]) : join(trace.tokens))
        @test trace.score == logsumexp(trace.scores .+ log.(probs))
        @test weight == 0.0

        # Constrained generation
        constraints = choicemap((:output, trace.output))
        new_trace, new_weight =
            generate(gpt3_mixture, (prompts, probs), constraints)

        @test new_trace.prompts == trace.prompts
        @test new_trace.probs == trace.probs
        @test new_trace.output == trace.output
        @test new_trace.tokens == trace.tokens
        @test isapprox(new_trace.score, trace.score, atol=1e-2)
        @test all(isapprox.(new_trace.scores, trace.scores, atol=1e-3))
        @test new_weight == new_trace.score
    end

    @testset "update" begin 
        # Generate initial trace
        prompts = ["What is the tallest mountain on Mars?",
                   "Which mountain do the Greek gods live upon?"]
        probs = [0.25, 0.75]
        output = "\n\nMount Olympus."
        constraints = choicemap((:output, output))
        trace, weight = generate(gpt3_mixture, (prompts, probs), constraints)

        # Updateless update
        argdiffs = (UnknownChange(), UnknownChange())
        new_trace, weight, retdiff, discard = 
            update(trace, get_args(trace), argdiffs, choicemap())
        @test new_trace == trace
        @test weight == 0.0
        @test retdiff isa NoChange
        @test isempty(discard)

        # Update output
        new_output = "\n\nThe mountain is Mount Olympus."
        constraints = choicemap((:output, new_output))
        argdiffs = (NoChange(), NoChange())
        new_trace, weight, retdiff, discard = 
            update(trace, get_args(trace), argdiffs, constraints)
        @test weight == get_score(new_trace) - get_score(trace)
        @test retdiff isa UnknownChange
        @test discard[:output] == output

        # Adjust prompts
        new_prompts = ["Which mountain do the Greek gods live upon?",
                       "What is the tallest mountain on Mars?"]
        argdiffs = (UnknownChange(), NoChange())        
        new_trace, weight, retdiff, discard = 
            update(trace, (new_prompts, probs), argdiffs, choicemap())
        @test isapprox(trace.scores, reverse(new_trace.scores), atol=1e-3)
        @test weight == get_score(new_trace) - get_score(trace)
        @test retdiff isa NoChange
        @test isempty(discard)

        # Adjust prompts and probabilities
        new_probs = [0.75, 0.25]
        argdiffs = (UnknownChange(), UnknownChange())
        new_trace, weight, retdiff, discard = 
            update(trace, (new_prompts, new_probs), argdiffs, choicemap())
        @test isapprox(trace.scores, reverse(new_trace.scores), atol=1e-3)
        @test isapprox(trace.score, new_trace.score, atol=1e-2)
        @test weight == get_score(new_trace) - get_score(trace)
        @test retdiff isa NoChange
        @test isempty(discard)
    end 

    @testset "regenerate" begin 
        # Generate initial trace
        prompts = ["What is the tallest mountain on Mars?",
                   "Which mountain do the Greek gods live upon?"]
        probs = [0.25, 0.75]
        output = "\n\nMount Olympus."
        constraints = choicemap((:output, output))
        trace, weight = generate(gpt3_mixture, (prompts, probs), constraints)

        # Regenerate nothing
        argdiffs = (UnknownChange(), UnknownChange())
        new_trace, weight, retdiff = 
            regenerate(trace, get_args(trace), argdiffs, select())
        @test new_trace == trace
        @test weight == 0.0
        @test retdiff isa NoChange

        # Regenerate output
        argdiffs = (NoChange(), NoChange())
        new_trace, weight, retdiff = 
            regenerate(trace, get_args(trace), argdiffs, select(:output))
        @test weight == 0.0
        @test retdiff isa UnknownChange

        # Adjust prompts
        new_prompts = ["Which mountain do the Greek gods live upon?",
                       "What is the tallest mountain on Mars?"]
        argdiffs = (UnknownChange(), NoChange())        
        new_trace, weight, retdiff = 
            regenerate(trace, (new_prompts, probs), argdiffs, select())
        @test isapprox(trace.scores, reverse(new_trace.scores), atol=1e-3)
        @test weight == get_score(new_trace) - get_score(trace)
        @test retdiff isa NoChange

        # Adjust prompts and probabilities
        new_probs = [0.75, 0.25]
        argdiffs = (UnknownChange(), UnknownChange())
        new_trace, weight, retdiff = 
            regenerate(trace, (new_prompts, new_probs), argdiffs, select())
        @test isapprox(trace.scores, reverse(new_trace.scores), atol=1e-3)
        @test isapprox(trace.score, new_trace.score, atol=1e-2)
        @test weight == get_score(new_trace) - get_score(trace)
        @test retdiff isa NoChange
    end

end
