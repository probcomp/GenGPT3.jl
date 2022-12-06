using Gen, GenGPT3

## Define a model that samples an intention, then an instruction 

"Samples a label uniformly at random."
@dist labeled_uniform(labels) =
    labels[uniform_discrete(1, length(labels))]

examples = """
Intention: (buy french-fries 1)
Instruction: I would like one order of french fries.
Intention: (buy lollipop 2)
Instruction: Could I have two lollipops please?
Intention: (buy coca-cola 3)
Instruction: Get me 3 cups of coke.
"""

gpt3 = GPT3GF(max_tokens=64, stop="\nIntention:")

"Sample an intention from a fixed set, then a natural language instruction."
@gen function instruction_model()
    intention ~ labeled_uniform([
        "(buy ice-cream 1)",
        "(buy ice-cream 2)",
        "(buy pizza 1)",
        "(buy pizza 2)"
    ])
    prompt = examples * "Intention: " * intention * "\nInstruction:"
    instruction ~ gpt3(prompt)
    return (intention, instruction)
end

instruction_model()

## Infer intentions given an instruction

"Print inferred probabilities for each instruction, given traces and weights."
function print_probs(traces, weights)
    probs = Dict{String, Float64}()
    for (tr, w) in zip(traces, weights)
        intention = tr[:intention]
        p = get(probs, intention, 0.0)
        probs[intention] = p + exp(w)
    end
    for (i, p) in probs
        print("Intention: ", i)
        println()
        println("Probability: ", round(p, digits=2))
    end    
end

instruction = " Get me two pizzas, pronto."
observations = choicemap((:instruction => :output, instruction))
traces, weights = importance_sampling(
    instruction_model, (), observations, 20
)
print_probs(traces, weights)

instruction = " I want some ice cream to share."
observations = choicemap((:instruction => :output, instruction))
traces, weights = importance_sampling(
    instruction_model, (), observations, 20
)
print_probs(traces, weights)

instruction = " I'm looking for something savory."
observations = choicemap((:instruction => :output, instruction))
traces, weights = importance_sampling(
    instruction_model, (), observations, 20
)
print_probs(traces, weights)
