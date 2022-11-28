# GenGPT3.jl

[GPT-3](https://en.wikipedia.org/wiki/GPT-3) as a generative function in [Gen.jl](https://www.gen.dev/), implemented by wrapping the [OpenAI API](https://openai.com/api/) in Gen's generative function interface.

## Usage

Install both Gen and this package via the Julia Pkg REPL:

```
add Gen
add https://github.com/probcomp/GenGPT3.jl.git
```

Add your OpenAI API key as an environment variable named `OPENAI_API_KEY`. You can follow [this guide](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety), or set `ENV["OPENAI_API_KEY"]` to the value of your API key in the Julia REPL.

Now you can construct GPT-3 as a generative function, and call GFI functions on it:

```julia
using Gen, GenGPT3

# Construct GPT3GenerativeFunction
gpt3 = GPT3GF(model="text-davinci-002", max_tokens=256)

# Untraced execution 
prompt = "What is the tallest mountain on Mars?"
output = gpt3(prompt)

# Traced execution
trace = simulate(gpt3, (prompt,))

# Constrained generation
constraints = choicemap((:output, "Olympus Mons."))
trace, weight = generate(gpt3, (prompt,), constraints)
```