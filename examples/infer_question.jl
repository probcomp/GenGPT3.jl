using Gen, GenGPT3

## Define a model that samples a question, then an answer 

gpt3 = GPT3GF(max_tokens=64)

"Samples a label uniformly at random."
@dist labeled_uniform(labels) =
    labels[uniform_discrete(1, length(labels))]

"Sample one of 3 questions, then sample an answer according to GPT-3."
@gen function qa_model_simple()
    question ~ labeled_uniform([
        "What is the tallest mountain on Mars?",
        "Which mountain do the Greek gods live upon?",
        "What is the tallest mountain in Greece?",
    ])
    prompt = question * "\n\n"
    answer ~ gpt3(prompt)
    return (question, answer)
end

## Infer the distribution over questions, given various answers

"Print inferred probabilities for each question, given traces and weights."
function print_probs(traces, weights)
    probs = Dict{String, Float64}()
    for (tr, w) in zip(traces, weights)
        question = tr[:question]
        p = get(probs, question, 0.0)
        probs[question] = p + exp(w)
    end
    for (qn, p) in probs
        print("Question: ")
        show(qn)
        println()
        println("Probability: ", round(p, digits=2))
    end    
end

answer = "Mount Olympus."
observations = choicemap((:answer => :output, answer))
traces, weights = importance_sampling(
    qa_model_simple, (), observations, 30
)
print_probs(traces, weights)

answer = "Mount Olympus is the tallest mountain."
observations = choicemap((:answer => :output, answer))
traces, weights = importance_sampling(
    qa_model_simple, (), observations, 30
)
print_probs(traces, weights)

answer = "Olympus Mons."
observations = choicemap((:answer => :output, answer))
traces, weights = importance_sampling(
    qa_model_simple, (), observations, 30
)
print_probs(traces, weights)

## Define a model with a more general question prior

"Sample a question and an answer according to GPT-3."
@gen function qa_model()
    question ~ gpt3("Question:")
    prompt = question * "\n\nAnswer:\n\n"
    answer ~ gpt3(prompt)
    return (question, answer)
end

## Define a targeted proposal to sample likely questions given answers

"Sample a likely question given answer."
@gen function qa_proposal(answer::String)
    prompt = ("Consider the following answer to a question:\n\n" * answer *
              "\n\nWhat question could have led to this answer?")
    question ~ gpt3(prompt)
    return question
end

## Use proposal for importance sampling of posterior over questions

answer = "Mount Olympus."
observations = choicemap((:answer => :output, answer))
traces, weights = importance_sampling(
    qa_model, (), observations, qa_proposal, (answer,), 30
)
print_probs(traces, weights)

answer = "Mount Olympus is the tallest mountain."
observations = choicemap((:answer => :output, answer))
traces, weights = importance_sampling(
    qa_model, (), observations, qa_proposal, (answer,), 30
)
print_probs(traces, weights)

answer = "Olympus Mons."
observations = choicemap((:answer => :output, answer))
traces, weights = importance_sampling(
    qa_model, (), observations, qa_proposal, (answer,), 30
)
print_probs(traces, weights)
