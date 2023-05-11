

function ppmi_vectors(a::Adam, words::Vector{String}, report_path::String)
    l = a.layers[1]
    h = l.semantic[1]

    file = open(joinpath(report_path, "ppmi_vectors.txt"), "w")

    for word in words

        word_string = map(x -> x[1], encode(word, a))

        if haskey(h.string_idx, word_string)
            
            wh = h.string_idx[word_string]

            ppmi_vector = head_vector(M(wh), h)

            msg_to_logs("$word\n", [file])

            for (ax, r) in ppmi_vector
                msg_to_logs("($ax, $r)\n", [file])
            end
            msg_to_logs("\n", [file])

        end
    end
    return nothing
end

function find_s_vector_synonyms(a::Adam, words::Vector{String}, report_path::String)
    l = a.layers[1]
    h = l.semantic[1]
    ppmi_vectors = map(wh -> norm_ppmi(head_vector(M(wh), h)), 1:h.D)
    map(word -> find_s_vector_synonyms(a, word, ppmi_vectors, report_path), words)
end

function find_s_vector_synonyms(a::Adam, word::String,
    ppmi_vectors::Vector{Vector{Tuple{M, R}}}, report_path::String)::Nothing

    synonyms_file = open(joinpath(report_path, "[$(word)]_vector_synonyms.csv"), "w")
    l = a.layers[1]
    h = l.semantic[1]

    word_string = map(x -> x[1], encode(word, a))

    if !haskey(h.string_idx, word_string)

        msg_to_logs("Слово не является мормемой первого слоя", [synonyms_file])

    else

        wh = h.string_idx[word_string]
        wh_vector = Dict(map(x -> x[1] => x[2], ppmi_vectors[wh]))
        wh_keys = Set(keys(wh_vector))

        distances = zeros(R, length(ppmi_vectors))

        for i ∈ eachindex(ppmi_vectors)
            if isempty(ppmi_vectors[i])
                distances[i] = sqrt(2.0)
                continue
            end

            i_vector = Dict(map(x -> x[1] => x[2], ppmi_vectors[i]))
            i_keys = Set(keys(i_vector))

            for key ∈ i_keys
                if key ∉ wh_keys
                    distances[i] += (i_vector[key] ^ 2)
                end
            end

            for key ∈ wh_keys
                if key ∉ i_keys
                    distances[i] += (wh_vector[key] ^ 2)
                end
            end

            for key ∈ intersect(wh_keys, i_keys)
                distances[i] += ((i_vector[key] - wh_vector[key]) ^ 2)
            end

            distances[i] = sqrt(distances[i])
        end

        synonyms = sortperm(distances)

        msg_to_logs("Слово ; Расстояние\n", [synonyms_file])

        for i ∈ synonyms
                
            msg_to_logs("$(decode(M(i), a)) ; " * @sprintf("%.4f\n", distances[i]),
                [synonyms_file])
        end
    end

    close(synonyms_file)

    return nothing
end

function find_d_vector_synonyms(a::Adam, words::Vector{String}, dims::Vector{Int}, report_path::String)
    l = a.layers[1]
    h = l.semantic[1]
    for dim ∈ dims
        l.Rnd = sign.(randn(dim, 2 * h.M))
        vectors = map(x -> zeros(F, dim), 1:h.D)
        @threads for wh ∈ 1:h.D 
            vectors[wh] = normalize(sparse2dense(head_vector(M(wh), h), l))
        end
        map(word -> find_d_vector_synonyms(a, word, vectors, dim, report_path), words)
    end
end

function find_d_vector_synonyms(a::Adam, word::String,
    vectors::Vector{Vector{F}}, dim::Int, report_path::String)::Nothing

    synonyms_file = open(joinpath(report_path, "[$(word)]_$(dim)_vector_synonyms.csv"), "w")
    l = a.layers[1]
    h = l.semantic[1]

    word_string = map(x -> x[1], encode(word, a))

    if !haskey(h.string_idx, word_string)

        msg_to_logs("Слово не является мормемой первого слоя", [synonyms_file])

    else

        wh = h.string_idx[word_string]
        vectors[wh]

        distances = zeros(R, length(vectors))

        for i ∈ eachindex(vectors)
            if vectors[i] == zeros(F, length(vectors[i]))
                distances[i] = 2.0
            else
                distances[i] = norm(vectors[i] - vectors[wh])
            end
        end

        synonyms = sortperm(distances)

        msg_to_logs("Слово ; Расстояние\n", [synonyms_file])

        for i ∈ synonyms
                
            msg_to_logs("$(decode(M(i), a)) ; " * @sprintf("%.4f\n", distances[i]),
                [synonyms_file])
        end
    end

    close(synonyms_file)

    return nothing
end

function count_equal(v1::Vector{M}, v2::Vector{M})::Int

    @assert length(v1) == length(v2)

    equal = 0

    for i ∈ 1:length(v1)

        (v1[i] == v2[i]) && (equal += 1)
        
    end

    return equal
end

find_synonyms(a::Adam, words::Vector{String}, report_path::String) =
    map(word -> find_synonyms(a, word, report_path), words)

function find_synonyms(a::Adam, word::String, report_path::String)::Nothing

    synonyms_file = open(joinpath(report_path, "[$(word)]_synonyms.csv"), "w")
    l = a.layers[2]

    wv = make_seed_morphemes!(a, 2, word)

    if length(wv) != 1

        msg_to_logs("Слово не является символом второго слоя", [synonyms_file])

    else

        s = wv[1]
        (i, m, w) = encode!(l, s)

        if isempty(w)
            
            msg_to_logs("Слово не является символом второго слоя", [synonyms_file])

        else

            msg_to_logs("Слово ; Часть совпавших кластеров\n", [synonyms_file])

            synonyms = map(m_ -> (m_, count_equal(w, l.morphemes[m_])),
                unique(l.symbol_morphemes))
            sort!(synonyms, by = x -> x[2], rev = true)

            for (m_, c) ∈ synonyms

                for s_ ∈ map(iv -> l.symbols[iv[1]], collect(l.sequences[m_]))
                    
                    msg_to_logs("$(decode(s_[1], a)) ; " * @sprintf("%.2f\n", c / l.N),
                        [synonyms_file])

                end
            end
        end
    end

    close(synonyms_file)

    return nothing
end

function find_full_synonyms(a::Adam, report_path::String, L::Int)::Nothing
    if L < 2
        return nothing
    elseif L == 2
        return find_full_synonyms(a, report_path)
    end

    full_synonyms = Int[]

    synonyms_file = open(joinpath(report_path, "full_synonyms_L$(L).txt"), "w")

    l = a.layers[L]

    for m ∈ unique(l.symbol_morphemes)

        w = l.morphemes[m]

        msg_to_logs("$w\n", [synonyms_file])

        push!(full_synonyms, length(l.sequences[m]))

        for s ∈ map(iv -> l.symbols[iv[1]], collect(l.sequences[m]))
                
            msg_to_logs("$(decode(s, L, a))\n", [synonyms_file])

        end

        msg_to_logs("\n", [synonyms_file])

    end

    return nothing
end

function find_full_synonyms(a::Adam, report_path::String)::Nothing

    synonyms_file = open(joinpath(report_path, "full_synonyms.txt"), "w")
    histogram_path = joinpath(report_path, "full_synonyms_histogram_.png")
    l = a.layers[2]

    full_synonyms = Int[]

    for m ∈ unique(l.symbol_morphemes)

        w = l.morphemes[m]

        msg_to_logs("$w\n", [synonyms_file])

        push!(full_synonyms, length(l.sequences[m]))

        for s ∈ map(iv -> l.symbols[iv[1]], collect(l.sequences[m]))
                
            msg_to_logs("$(decode(s[1], a)) $(a.app["sv_length"][s[1]])\n", [synonyms_file])

        end

        msg_to_logs("\n", [synonyms_file])

    end

    for h ∈ l.semantic
        find_full_synonyms(a, h, report_path)
    end

    histogram(log.(2, full_synonyms), yaxis=(:log10))
    title!("Full synonyms")
    xlabel!("Synonyms (log2)")
    ylabel!("Morphemes")
    
    savefig(histogram_path)

    return nothing
end

function find_full_synonyms(a::Adam, h::Head, report_path::String)::Nothing

    synonyms_file = open(joinpath(report_path, "full_synonyms_h$(h.N).txt"), "w")
    s_histogram_path = joinpath(report_path, "symbols_histogram_h$(h.N).png")
    w_histogram_path = joinpath(report_path, "weights_histogram_h$(h.N).png")
    l = a.layers[2]

    lengths = map(wh -> mapreduce(m -> length(l.sequences[m]), +, l.semantic[h.N].m̄[wh]), 1:h.k)
    weights = zeros(h.k)

    if !haskey(a.app, "weights")
        a.app["weights"] = map(s -> F(1.0), l.symbols)
    end

    for wh ∈ sortperm(lengths)

        msg_to_logs("wh=$wh: ", [synonyms_file])

        for m ∈ l.semantic[h.N].m̄[wh]

            for s ∈ map(iv -> l.symbols[iv[1]], collect(l.sequences[m]))

                weights[wh] += a.app["weights"][s]                
                msg_to_logs("$(decode(s[1], a)) ; ", [synonyms_file])
    
            end
        
        end

        msg_to_logs("\n\n", [synonyms_file])

    end

    histogram(log.(2, lengths))
    title!("Clusters population")
    xlabel!("Population (log2)")
    ylabel!("Clusters")

    savefig(s_histogram_path)

    histogram(log.(2, weights))
    title!("Clusters weights")
    xlabel!("Weight (log2)")
    ylabel!("Clusters")

    savefig(w_histogram_path)

    return nothing
end

function find_same_phrases(a::Adam, report_path::String)::Nothing

    phrases_file = open(joinpath(report_path, "phrases.txt"), "w")
    l = a.layers[2]

    for m ∈ 1:l.n

        w = l.morphemes[m]

        msg_to_logs("$w\n", [phrases_file])

        for indices ∈ collect(l.sequences[m])

            for i ∈ indices

                s = l.symbols[i]

                msg_to_logs("$(decode(s[1], a))", [phrases_file])

            end

            msg_to_logs("\n", [phrases_file])

        end

        msg_to_logs("\n", [phrases_file])

    end

    return nothing
end



function norm_ppmi(head_vector::Vector{Tuple{M, R}})::Vector{Tuple{M, R}}
    if isempty(head_vector)
        return head_vector
    end
    v_norm = sqrt(mapreduce(x -> x[2]^2, +, head_vector))
    return map(x -> (x[1], x[2] / v_norm), head_vector)
end