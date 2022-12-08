module GenGPT3

using Gen
using JSON3
using HTTP
using Base: @kwdef

import Gen:
    get_choices, get_args, get_retval, get_score, get_gen_fn,
    simulate, generate, project, assess, update, regenerate

export GPT3ChoiceMap, GPT3Trace, GPT3GenerativeFunction, GPT3GF

include("tokenizer.jl")
include("web_api.jl")
include("gen_fn.jl")

end
