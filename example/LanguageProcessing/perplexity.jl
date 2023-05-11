

function parse_morpheme!(l::Layer, i::I, w::Morpheme, r::R, t::T=T(0)
                      )::Tuple{Vector{M}, R, T}
    p = l.parser
    s = l.parser.stack

    # Find the winner: morpheme pair with max probability
    winners = sortperm(p.r, rev = true)
    p.r[end] = get_count(s[end][1], w, l)
    i_max = winners[1]
    r_max = p.r[i_max]
    for k ∈ 2:p.O
        if mergeable(l, i_max) || r_max == R(0.)
            break
        end
        i_max = winners[k]
        r_max = p.r[i_max]
    end
    if (isempty(s[i_max+1][1]) || isempty(s[i_max][1]))
        update_stack_!(l, 1, w, r, t, [i])
        return p.out
    end

    # Try to merge the winner morpheme pair
    s̄::Vector{I} = [s[i_max][4]; s[i_max+1][4]]  # layer symbol index string
    (w_, r_, v_) = merge_morphemes(l, s[i_max][1], s[i_max+1][1],
                                       s[i_max][2], s[i_max+1][2], s̄)
    if (isempty(w_))  # Merge failed
        update_stack_!(l, 1, w, r, t, [i])
        p.last_merge = 0
        p.last_merged_morpheme = (M[], R(0.), T(0))
        return p.out
    end

    l.opt["merge_surprise"] += get_merge_surprise(s[i_max][1], s[i_max+1][1], l)
    # Replace the merged morphemes in the stack
    s[i_max] = (w_, r_, s[i_max+1][3], s̄)
    p.last_merge = i_max
    p.last_merged_morpheme = s[i_max][1:3]
    update_stack_!(l, i_max+1, w, r, t, [i])
    p.r[i_max] = get_count(s[i_max][1], s[i_max+1][1], l)

    # Update R matrix for the predecessor of the merged morpheme
    (i_max > 1) && (p.r[i_max-1] = get_count(s[i_max-1][1], s[i_max][1], l))

    return (M[], R(0.), T(0))
end

""" Update parser stack: push new morpheme, update pair count """
function update_stack_!(l::Layer, i::Int, w::Morpheme, r::R, t::T, s̄::Vector{I})::Nothing
    p = l.parser
    s = l.parser.stack
    p.out = (i==1) ? s[1][1:3] : (M[], R(0.), T(0))
    i < l.parser.O ? (
        s[i:(end-1)] = s[(i+1):end];
        p.r[i:(end-1)] = p.r[(i+1):end]
    ) : (i == l.parser.O) && (s[end-1] = s[end])
    s[end] = (w, r, t, s̄)
    p.r[end] = typemin(R)
    p.r[end-1] = get_count(s[end-1][1], w, l)
    return nothing
end

""" Create a new morpheme in a layer merging two layer morphemes """
function merge_morphemes(l::Layer, w1::Morpheme, w2::Morpheme, r1::R, r2::R,
                          s̄::Vector{I})::Tuple{Morpheme, R, R, R}  # w, reward, value, probability
    # Check if the merged morpheme ww ← w1&w2 already exists
    ww::Morpheme = map(h -> get(h.w, (w1[h.N], w2[h.N]), M(0)), l.semantic)
    if (haskey(l.index, ww))
        m = l.index[ww]
        if (s̄ ∉ l.sequences[m])
            push!(l.sequences[m], s̄)
        end
        return (ww, r1+r2, get_value(ww, l), get_count(w1, w2, l))
    end

    return (M[], R(0.), R(0.), R(0.))
end

function get_merge_surprise(w::Morpheme, w′::Morpheme, l::Layer)::R
    if invalid(w) || invalid(w′)
        return log(R(1.0 / l.semantic[1].R_if))
    end
    #return mapreduce(h -> log(get(h.R[w[h.N]], w′[h.N], R(1.)) / h.R_if), +, l.semantic) / l.N
    return mapreduce(h -> log(get(h.R[w[h.N]], w′[h.N], R(1.)) / h.R_i[w[h.N]]), +, l.semantic) / l.N
end

function get_transition_surprise(w::Morpheme, w′::Morpheme, l::Layer)::R
    if invalid(w) || invalid(w′)
        #return log(R(1.0 / l.semantic[1].R_if))
        #return log(R(1.0 / l.semantic[1].D))
        return - log(R(length(l.symbols)))
    end
    #return mapreduce(h -> log(get(h.R[w[h.N]], w′[h.N], R(1.)) / h.R_if), +, l.semantic) / l.N
    if any(map(h -> h.R_i[w[h.N]] == R(0.0), l.semantic))
        return - log(R(length(l.symbols)))
    end
    return (mapreduce(h -> log(get(h.R[w[h.N]], w′[h.N], R(1.)) / h.R_i[w[h.N]]), +, l.semantic)) / l.N - log(length(l.sequences[l.index[w′]]))
end

function get_morpheme_surprise(w::Morpheme, l::Layer)::R
    if invalid(w)
        return - log(R(l.semantic[1].R_if))
    end
    return (mapreduce(h -> log(max(R(1.0), h.R_i[w[h.N]]) / h.R_if), +, l.semantic)) / l.N - log(length(l.sequences[l.index[w]]))
end

function PPW(a::Adam, text::String, report_path::String)::Vector{R}

    report_file = open(joinpath(report_path, "PPW_report.txt"), "w")

    words_v = lowercase.(split(text, " "))

    N = length(words_v)

    mv = text_reparsing_history!(a, text)
    PPL = zeros(R, a.L)

    synonym_surprise = R(0.0)

    for L in 2:a.L

        lp = a.layers[L-1]
        l = a.layers[L]

        l.opt["merge_surprise"] = 0.0

        r = R(0.0)
        t = T(1)

        r_ = R(0.0)
        t_ = t

        clear_parser_!(l)

        m_history = I[]
        sizehint!(m_history, length(mv))

        for mp ∈ mv

            if mp <= I(0)
                #w_, r_, t_ = parse_input!(l, I(0), M[], r, T(mp))
                w_, r_, t_ = parse_morpheme!(l, I(0), M[], r, T(mp))
                if !isempty(w_)
                    push!(m_history, l.index[w_])
                elseif t_ < T(-L)
                    push!(m_history, I(t_))
                end                     
            else
                s = lp.morphemes[mp]
                
                (i, m, w) = encode!(l, s)    
                #w_, r_, t_ = parse_input!(l, i, w, r, t)       
                #w_, r_, t_ = parse_input!(l, i, w, r, i == I(0) ? T(-100) : t) #temp
                w_, r_, t_ = parse_morpheme!(l, i, w, r, i == I(0) ? T(-100) : t)
                
                if !isempty(w_)
                    push!(m_history, l.index[w_])
                elseif t_ < T(-L) #|| t_ == T(0) #temp
                    push!(m_history, I(t_))
                end
            end
        end

        W_, R_, T_ = clear_parser_!(l)
        for (i, w) ∈ enumerate(W_)
            t_ = T_[i]
            if !isempty(w)
                push!(m_history, l.index[w])
            elseif t_ < T(-L) || #t_ == T(0) #temp
                push!(m_history, I(t_))
            end 
        end

        clean_history = filter(m -> m > I(0), m_history)

        synonym_surprise += mapreduce(m -> -log(length(l.sequences[m])), +, clean_history)

        transition_surprise = mapreduce(i -> get_transition_surprise(
            l.morphemes[clean_history[i]], l.morphemes[clean_history[i + 1]], l), +, 1:(length(clean_history)-1))
        
        morphemes_surprise = mapreduce(m -> get_morpheme_surprise(l.morphemes[m], l), +, clean_history)

        #PPL[L] = exp((l.opt["merge_surprise"] + transition_surprise)/ N) 
        PPL[L] = exp(-(transition_surprise + synonym_surprise) / N)

        msg_to_logs(string("L$(L) PPW = $(PPL[L])\nN = $(N)\navg. transition surprise = $(transition_surprise / N)\n",
        "avg. merge surprise = $(l.opt["merge_surprise"] / N)\navg. morpheme surprise = $(morphemes_surprise / N)\n"), [report_file])

        mv = m_history
    end
    
    return PPL
end

function define_probabilities(l::Layer)::Vector{Dict{I, R}}

    probabilities = fill(Dict{I, R}(), l.n)
    w = fill(M(0), l.N)
    w_ = fill(M(0), l.N)
    for m in l.n

        w = l.morpheme[m]

        norm = 0.0

        for m_ in l.memory[m]

            w_ = l.morpheme[m_]

            probability = exp(mapreduce(h -> log(get(h.R[w[h.N]], w_[h.N], R(1.0)) / max(h.R_i[w[h.N]], R(1.0))), +, l.semantic) / l.N)

            probabilities[m][m_] = probability
            
            norm += probability * length(l.sequences[m])
        end

        for m_ in l.memory[m]

            probabilities[m][m_] = probabilities[m][m_] / norm

        end

    end

    return probabilities
end         
            
function clear_parser_!(l::Layer)::Tuple{Vector{Morpheme}, Vector{R}, Vector{T}}
    wv::Vector{Morpheme} = Morpheme[]
    rv::Vector{R} = R[]
    tv::Vector{T} = T[]
    for _ ∈ 1 : l.parser.O + 2
        (w, r, t) = parse_morpheme!(l, I(0), M[], R(0.), T(0))
        (!isempty(w) && ∉(M(0), w)) && (
            push!(wv, w); push!(rv, r); push!(tv, t))
    end
    l.parser.last_merge = 0
    l.parser.last_merged_morpheme = (M[], R(0.), T(0))
    clear_parser_memory!(l.parser)
    return (wv, rv, tv)
end

