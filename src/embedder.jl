export findsimilar

## Embedder ##

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

## EmbeddingStore ##

"""
    EmbeddingStore

Data structure that stores a list of texts, their vector embeddings, and
associated metadata. Texts with similar embeddings to an input text can be
found using the `findsimilar` function. New texts and embeddings can be added
with the `append!` and `push!` functions.

Loading and saving as a JSON file can be done via the `load_store` and
`save_store` functions. In addition, `EmbeddingStore` implements the `Tables.jl`
interface, so it can be used with `Tables.jl` compatible file formats.
"""
struct EmbeddingStore
    embedder::Embedder
    data::FlexTable{1}
end

"""
    EmbeddingStore(embedder = Embedder())

Constructs an empty `EmbeddingStore` with the specified `embedder`.
"""
function EmbeddingStore()
    return EmbeddingStore(Embedder())
end

function EmbeddingStore(embedder::Embedder)
    data = FlexTable(text=String[], embedding=Vector{Float64}[])
    return EmbeddingStore(embedder, data)
end

"""
    EmbeddingStore([embedder], texts, [embeddings, metadata]; columns...)

Constructs an `EmbeddingStore` with the specified `embedder`, `texts` (as a 
vector of strings), `embeddings` (as a vector of vectors of floats), and
optional `metadata` (as a vector of dictionaries). Additional metadata columns
can be specified as keyword arguments, with keywords serving as column names.
"""
function EmbeddingStore(
    embedder::Embedder,
    texts::AbstractVector{<:AbstractString},
    embeddings::AbstractVector{<:AbstractVector{<:Real}} = embedder(texts),
    metadata::Union{AbstractVector, Nothing} = nothing;
    columns...
)
    store = EmbeddingStore(embedder)
    append!(store, texts, embeddings, metadata)
    for (key, col) in columns
        @assert length(texts) == length(col)
        valtype = eltype(col)
        col = convert(Vector{Union{valtype, Missing}}, col)
        setproperty!(store.data, key, col)
    end
    return store
end

function EmbeddingStore(
    texts::AbstractVector{<:AbstractString},
    embeddings::AbstractVector{<:AbstractVector{<:Real}},
    metadata::Union{AbstractVector, Nothing} = nothing;
    columns...
)
    return EmbeddingStore(Embedder(), texts, embeddings, metadata; columns...)
end

function EmbeddingStore(texts::AbstractVector{<:AbstractString}; columns...)
    return EmbeddingStore(Embedder(), texts; columns...)
end

Base.getindex(store::EmbeddingStore, i::Integer) = store.data[i]
Base.length(store::EmbeddingStore) = length(store.data)

"""
    append!(store::EmbeddingStore, texts, [embeddings, metadata]; columns...)

Appends `texts` (as a vector of strings), `embeddings` (as a vector of vectors
of floats), and optional `metadata` (as a vector of dictionaries) to the
`EmbeddingStore`. Additional metadata columns can be specified as keyword
arguments, with keywords serving as column names.
"""
function Base.append!(
    store::EmbeddingStore,
    texts::AbstractVector{<:AbstractString},
    embeddings::AbstractVector{<:AbstractVector{<:Real}} = store.embedder(texts),
    metadata::Union{AbstractVector, Nothing} = nothing;
    columns...
)
    nrows_orig = length(store.data)
    # Append texts and embeddings
    @assert length(texts) == length(embeddings)
    texts = convert(Vector{String}, texts)
    embeddings = convert(Vector{Vector{Float64}}, embeddings)
    append!(store.data.text, texts)
    append!(store.data.embedding, embeddings)
    # Append metadata if provided
    if !isnothing(metadata)
        @assert length(texts) == length(metadata)
        colnames = collect(columnnames(store.data))
        for (i, row) in enumerate(metadata)
            for (key, val) in row
                valtype = typeof(val)
                nrows = length(store.data)
                if !(key in colnames)
                    # Add column if it doesn't exist
                    column = Vector{Union{valtype, Missing}}(missing, nrows)
                    push!(colnames, key)
                    setproperty!(store.data, key, column)
                elseif !(valtype <: eltype(store.data.key))
                    # Promote column type if necessary
                    valtype = Union{valtype, eltype(store.data.key)}
                    column = Vector{Union{valtype, Missing}}(missing, nrows)
                    setproperty!(store.data, key, column)
                else
                    # Look-up column
                    column = getproperty(store.data, key)
                end
                # Set value
                column[i + nrows_orig] = val
            end
        end
    end
    # Add extra columns
    for (key, col) in columns
        @assert length(texts) == length(col)
        if !(key in columnnames(store.data))
            valtype = eltype(col)
            new_col = Vector{Union{valtype, Missing}}(missing, nrows_orig)
            setproperty!(store.data, key, new_col)
        end
        append!(getproperty(store.data, key), col)
    end
    # Ensure all columns have the same length
    nrows = length(store.data.text)
    for key in columnnames(store.data)
        column = getproperty(store.data, key)
        if length(column) < nrows
            append!(column, fill(missing, nrows - length(column)))
        end
    end
    return store
end

"""
    push!(store::EmbeddingStore, text, [embedding, metadata]; kwargs...)

Adds a `text` (as a string), `embedding` (as a vector of floats), and optional
`metadata` (as a dictionary) to the `EmbeddingStore`. Additional metadata
can be specified as keyword arguments, with keywords serving as field names.
"""
function Base.push!(
    store::EmbeddingStore,
    text::AbstractString,
    embedding::AbstractVector{<:Real} = store.embedder(text),
    metadata::Union{AbstractDict, Nothing} = nothing;
    kwargs...
)
    # Add text and embedding
    nrows = length(store.data)
    text = convert(String, text)
    embedding = convert(Vector{Float64}, embedding)
    push!(store.data.text, text)
    push!(store.data.embedding, embedding)
    # Add metadata
    metadata = isnothing(metadata) ? kwargs : merge!(metadata, kwargs)
    for (key, val) in metadata
        if !(key in columnnames(store.data))
            valtype = typeof(val)
            column = Vector{Union{valtype, Missing}}(missing, nrows)
            setproperty!(store.data, key, column)
        else 
            column = getproperty(store.data, key)
        end
        push!(column, val)
    end
    # Ensure all columns have the same length
    nrows = length(store.data.text)
    for key in columnnames(store.data)
        column = getproperty(store.data, key)
        if length(column) < nrows
            append!(column, fill(missing, nrows - length(column)))
        end
    end
    return store
end

"""
    findsimilar(store::EmbeddingStore, text, k; reversed=false)
    findsimilar(store::EmbeddingStore, embedding, k; reversed=false)

Returns the `k` most similar entries to the `text` or `embedding` in the
`EmbeddingStore`. If `reversed` is `true`, entries are returned in order of
increasing similarity.
"""
function findsimilar(
    store::EmbeddingStore, text::AbstractString, k::Int;
    reversed::Bool = false
)
    embedding = store.embedder(text)
    return findsimilar(store, embedding, k; reversed=reversed)
end

function findsimilar(
    store::EmbeddingStore, embedding::AbstractVector{<:Real}, k::Int;
    reversed::Bool = false
)
    sims = similarity(embedding, store.data.embedding)
    k = min(k, length(store.data))
    idxs = partialsortperm(sims, 1:k, rev=true)
    return reversed ? reverse(store.data[idxs]) : store.data[idxs]
end

"""
    findsimilar(filter_fn::Function,
                store::EmbeddingStore, text, k; reversed=false)
    findsimilar(filter_fn::Function,
                store::EmbeddingStore, embedding, k; reversed=false)

Returns the `k` most similar entries to the `text` or `embedding` in the
`EmbeddingStore` after filtering entries with `filter_fn`. If `reversed` is
`true`, entries are returned in order of increasing similarity.
"""
function findsimilar(
    filter_fn::Function,
    store::EmbeddingStore, text::AbstractString, k::Int;
    reversed::Bool = false
)
    embedding = store.embedder(text)
    return findsimilar(filter_fn, store, embedding, k; reversed=reversed)
end

function findsimilar(
    filter_fn::Function, 
    store::EmbeddingStore, embedding::AbstractVector{<:Real}, k::Int;
    reversed::Bool = false
)
    data = filter(filter_fn, store.data)
    sims = similarity(embedding, data.embedding)
    k = min(k, length(data))
    idxs = partialsortperm(sims, 1:k, rev=true)
    return reversed ? reverse(data[idxs]) : data[idxs]
end

# Tables.jl integration

Tables.rows(store::EmbeddingStore) = Tables.rows(store.data)
Tables.columns(store::EmbeddingStore) = Tables.columns(store.data)
Tables.istable(::Type{<:EmbeddingStore}) = true
Tables.rowaccess(::Type{<:EmbeddingStore}) = true
Tables.columnaccess(::Type{<:EmbeddingStore}) = true
Tables.schema(store::EmbeddingStore) = Tables.schema(store.data)

EmbeddingStore(columns::Tables.AbstractColumns) =
    EmbeddingStore(Embedder(), FlexTable(columns))

# Loading and saving

"""
    save_store(path::AbstractString, store::EmbeddingStore)

Save an `EmbeddingStore` to `path` as a JSON file.
"""
function save_store(path::AbstractString, store::EmbeddingStore)
    json = arraytable(store.data)
    open(path, "w") do io
        JSON3.pretty(io, json)
    end
    return path
end

"""
    load_store(path::AbstractString, [embedder]; types...)

Load an `EmbeddingStore` from `path` as a JSON file, optionally specifying the
`embedder` to use, and column types as keyword arguments.
"""
function load_store(path::AbstractString, embedder=Embedder(); types...)
    json = jsontable(read(path, String))
    data = FlexTable(json)
    data.text = convert(Vector{String}, data.text)
    data.embedding = convert(Vector{Vector{Float64}}, data.embedding)
    for (key, ty) in types
        ty = Union{ty, Missing}
        setproperty!(data, key, convert(Vector{ty}, getproperty(data, key)))
    end
    return EmbeddingStore(embedder, data)
end
