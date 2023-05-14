
"""
    Embedder(
        model::String = "text-embedding-ada-002",
        api_key_lookup::Function = lookup_openai_api_key,
        organization_lookup::Function = lookup_openai_organization
    )

Constructs an `Embedder` object that can be used to embed text using the
OpenAI Embeddings API. Once constructed, an `embedder` can be called with a 
string or vector of strings to embed the text using the specified model.
Token IDs can also be provided instead of strings.
"""
@kwdef struct Embedder <: Function
    model::String = "text-embedding-ada-002"
    api_key_lookup::Function = lookup_openai_api_key
    organization_lookup::Function = lookup_openai_organization
end

function (embedder::Embedder)(
    input::Union{AbstractString, AbstractVector{<:Integer}}
)
    response = embeddings_api_call(
        input; model = embedder.model, 
        api_key= embedder.api_key_lookup(),
        organization = embedder.organization_lookup()
    )
    return collect(Float64, response.data[1].embedding)
end

function (embedder::Embedder)(
    input::Union{AbstractVector{<:AbstractString}, 
                 AbstractVector{<:AbstractVector{<:Integer}}}
)
    response = embeddings_api_call(
        input; model = embedder.model, 
        api_key= embedder.api_key_lookup(),
        organization = embedder.organization_lookup()
    )
    @assert length(response.data) == length(input)
    embeddings = Vector{Vector{Float64}}(undef, length(input))
    for result in response.data
        embeddings[result.index + 1] = collect(Float64, result.embedding)
    end
    return embeddings
end

"""
    similarity(e1, e2)

Compute the cosine similarity between two embeddings.
"""
function similarity(e1::AbstractVector{<:Real}, e2::AbstractVector{<:Real})
    sim = e1' * e2 / (sqrt(e1' * e1) * sqrt(e2' * e2))
    return max(min(sim, 1.0), -1.0)
end

"""
    similarity(e, embeddings)

Compute the cosine similarity an embedding and a list of embeddings.
"""
function similarity(
    e::AbstractVector{<:Real},
    embeddings::AbstractVector{<:AbstractVector{<:Real}}
)
    sims = [similarity(e, e2) for e2 in embeddings]
    return sims
end

function similarity(
    e::AbstractVector{<:Real},
    embeddings::AbstractMatrix{<:Real}
)
    sims = [similarity(e, e2) for e2 in eachcol(embeddings)]
    return sims
end
