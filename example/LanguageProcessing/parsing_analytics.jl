

function find_words(text::String,
    popular_threshold::Int)::Tuple{Dict{String, Int64}, Dict{String, Int}}

    words_v = lowercase.(split(text, " "))
    words::Dict{String, I} = countmap(words_v)
    popular_words = filter(x -> x[2] > popular_threshold, words)
    return (words, popular_words)
end

function learning_result(learned_morphemes::Vector{String},
    target_vocabulary::Dict{String, Int})::Tuple{Dict, Dict}

    result = Dict{String, Any}()

    result["total words"] = mapreduce(
        wc -> wc[2], +, target_vocabulary, init=0)
    
    result["unique words"] = length(target_vocabulary)
    result["unique words learned"] = 0
    result["total words learned"] = 0
    words_learned = Dict{String, Int}()

    for word_ ∈ learned_morphemes
        if last(word_, 1) == "_"
            word = chop(word_)
            if haskey(target_vocabulary, word)
                words_learned[word_] = target_vocabulary[word]
                result["unique words learned"] += 1
                result["total words learned"] += target_vocabulary[word]
            end
        end
    end
    
    result["unique words learned percent"] = result["unique words"] != 0 ? 
        result["unique words learned"] / result["unique words"] : -1.0

    result["total words learned percent"] = result["total words"] != 0 ?
        result["total words learned"] / result["total words"] : -1.0

    return (result, words_learned)
end

function text_parsing_results(a::Adam, text::String)::Dict{String, Any}
    l = a.layers[1]
    h = l.semantic[1]

    popular_threshold = get(a.app, "popular_threshold", 2.0)
    all_words, popular_words = find_words(
        text, round(Int, popular_threshold * h.semantic_threshold))

    adam_words = map(w -> decode(h.string[w], a), 1:h.D)

    result, words_learned = learning_result(
        adam_words, all_words)

    popular_result, popular_words_learned = learning_result(
        adam_words, popular_words)

    a.app["1_st_layer_parsing"] = Dict()
    a.app["1_st_layer_parsing"]["results"] = result

    for (key, value) ∈ popular_result
        a.app["1_st_layer_parsing"]["results"][key * " (popular)"] = value
    end

    a.app["1_st_layer_parsing"]["results"]["morphemes formed"] = h.D

    a.app["1_st_layer_parsing"]["words learned"] = words_learned
    a.app["1_st_layer_parsing"]["words learned (popular)"] =
        popular_words_learned

    # morpheme -> one pair (of possibly several)
    separations = Dict(w => ww for (ww, w) in h.w)

    a.app["1_st_layer_parsing"]["morphemes_separations"] = separations        
    a.app["1_st_layer_parsing"]["words_separations"] = Dict(
        decode(M(w), a) => separate_morpheme(a, M(w), separations) for w in 1:h.D)

    return a.app["1_st_layer_parsing"]["results"]
end 

function separate_morpheme(a::Adam, w::M, separations::Dict)::String
    w ∉ keys(separations) && return decode(w, a, 1)

    w1, w2 = separations[w]

    return "[" * decode(w1, a, 1) * "][" * decode(w2, a, 1) * "]"
end