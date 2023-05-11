

function filter_sample!(mv::Vector{I}, a::Adam, L::Int)::Nothing
    @assert L >= 2
    if L > 3
        return nothing
    end
    lp = a.layers[L-1]
    l = a.layers[L]

    symbols = Set(keys(l.symbol_index))

    mv_filtered = I[]
    sizehint!(mv_filtered, length(mv))

    i_last = 0
    faulty = false
    for (i, m) in enumerate(mv)
        if m < I(0)
            if faulty
                push!(mv_filtered, I(0))
            else
                append!(mv_filtered, view(mv, i_last+1:i-1))
            end
            push!(mv_filtered, m)
            i_last = i            
            faulty = false
        else
            if m == I(0) || lp.morphemes[m] ∉ symbols
                faulty = true
            end
        end
    end
    mv = mv_filtered
    return nothing
end
function train_layer!(l::Layer, es::Vector{Morpheme},
    er::Vector{R}, et::Vector{T}, need_episodic::Bool=false, logs=[])::Nothing

    t_0 = Dates.now()
    msg_to_logs(string("training layer #$(l.L) started: ",
    Dates.format(t_0, "yyyy-mm-dd HH:MM:SS"), "\n"), logs)

    for (j, s) ∈ enumerate(es)
        
        (i, m, w) = encode!(l, s)            
        parse_input!(l, i, w, er[j], et[j])
        need_episodic && fill_episodic!(l.episodic, i, w, er[j], et[j])
        
    end

    t_1 = Dates.now()
    msg_to_logs(string("training layer #$(l.L) completed: ",
    Dates.format(t_1, "yyyy-mm-dd HH:MM:SS"),
    " ($(time_interval(t_0, t_1)))\n\n"), logs)
    
    return nothing        
end

function train_layer!(a::Adam, L::Int, mv::Vector{I}, logs=[])::Nothing

    @assert L >= 2
    t_0 = Dates.now()
    msg_to_logs(string("training layer #$L started: ",
    Dates.format(t_0, "yyyy-mm-dd HH:MM:SS"), "\n"), logs)

    lp = a.layers[L-1]
    l = a.layers[L]

    r = R(0.0)
    t = T(1)

    prev_mp = I(0)

    last_time = Dates.now()
    chunk = 10_000_000
    counter = 0

    for mp ∈ mv
        if mp <= I(0)
            if prev_mp > I(0)
                parse_input!(l, I(0), M[], r, T(mp))
            end
            prev_mp = mp
            counter += 1
            continue
        end
        s = lp.morphemes[mp]
        
        (i, m, w) = encode!(l, s)            
        parse_input!(l, i, w, r, t)       
        
        prev_mp = mp

        counter += 1
        if counter % chunk == 0
            new_time = Dates.now()
            msg_to_logs(string("$(counter) symbols parsed: ",
            Dates.format(new_time, "yyyy-mm-dd HH:MM:SS"),
            " (last chunk = $(chunk) for $(time_interval(last_time, new_time)))\n"), logs)
            last_time = new_time
        end
    end

    t_1 = Dates.now()
    msg_to_logs(string("training layer #$L completed: ",
    Dates.format(t_1, "yyyy-mm-dd HH:MM:SS"),
    " ($(time_interval(t_0, t_1)))\n\n"), logs)
    
    return nothing        
end

function reparse_morphemes!(a::Adam, L::Int, mv::Vector{I}, logs=[])::Vector{I}

    t_0 = Dates.now()
    msg_to_logs(string("reparsing layer #$L started: ",
    Dates.format(t_0, "yyyy-mm-dd HH:MM:SS"), "\n"), logs)

    lp = a.layers[L-1]
    l = a.layers[L]

    r = R(0.0)
    t = T(1)

    r_ = R(0.0)
    t_ = t

    clear_parser!(l)

    m_history = I[]
    sizehint!(m_history, length(mv))

    last_time = Dates.now()
    chunk = 10_000_000
    counter = 0

    for mp ∈ mv

        if mp <= I(0)
            #w_, r_, t_ = parse_input!(l, I(0), M[], r, T(mp))
            w_, r_, t_ = parse_input!(l, I(0), M[], r, mp == I(0) ? T(-100) : T(mp)) #temp
            if !isempty(w_)
                push!(m_history, l.index[w_])
            elseif t_ < T(-L) #|| t_ == T(0) #temp
                push!(m_history, I(t_))
            end                     
        else
            s = lp.morphemes[mp]
            
            (i, m, w) = encode!(l, s)    
            #w_, r_, t_ = parse_input!(l, i, w, r, t)       
            w_, r_, t_ = parse_input!(l, i, w, r, i == I(0) ? T(-100) : t) #temp
            
            if !isempty(w_)
                push!(m_history, l.index[w_])
            elseif t_ < T(-L) #|| t_ == T(0) #temp
                push!(m_history, I(t_))
            end
        end

        counter += 1
        if counter % chunk == 0
            new_time = Dates.now()
            msg_to_logs(string("$(counter) symbols reparsed: ",
            Dates.format(new_time, "yyyy-mm-dd HH:MM:SS"),
            " (last chunk = $(chunk) for $(time_interval(last_time, new_time)))\n"), logs)
            last_time = new_time
        end
    end

    W_, R_, T_ = clear_parser!(l)
    for (i, w) ∈ enumerate(W_)
        t_ = T_[i]
        if !isempty(w)
            push!(m_history, l.index[w])
        elseif t_ < T(-L) || #t_ == T(0) #temp
            push!(m_history, I(t_))
        end 
    end

    t_1 = Dates.now()
    msg_to_logs(string("reparsing layer #$L completed: ",
    Dates.format(t_1, "yyyy-mm-dd HH:MM:SS"),
    " ($(time_interval(t_0, t_1)))\n\n"), logs)
    
    return m_history       
end

function reparse_morphemes!(l::Layer, es::Vector{Morpheme}, er::Vector{R},
    et::Vector{T}, logs=[])::Tuple{Vector{Morpheme}, Vector{R}, Vector{T}}

    t_0 = Dates.now()
    msg_to_logs(string("reparsing layer #$(l.L) started: ",
    Dates.format(t_0, "yyyy-mm-dd HH:MM:SS"), "\n"), logs)

    clear_parser!(l)

    wv = Morpheme[]
    rv = R[]
    tv = T[]

    for (j, s) ∈ enumerate(es)
        
        (i, m, w) = encode!(l, s)            
        w_, r_, t_ = parse_input!(l, i, w, er[j], et[j])
        !isempty(w_) && (push!(wv, w_); push!(rv, r_); push!(tv, t_))

    end

    (W_, R_, T_) = clear_parser!(l)
    append!(wv, W_)
    append!(rv, R_)
    append!(tv, T_)

    t_1 = Dates.now()
    msg_to_logs(string("reparsing layer #$(l.L) completed: ",
    Dates.format(t_1, "yyyy-mm-dd HH:MM:SS"),
    " ($(time_interval(t_0, t_1)))\n\n"), logs)
    
    return (wv, rv, tv)       
end

function expand_layer_morphemes!(l::Layer, n::Int)::Nothing

    if n <= l.W
        return nothing
    end

        # Update layer memory (created empty)
    append!(l.memory, [Set(I[]) for _ ∈ l.W + 1 : n])
    append!(l.morphemes, [M[] for _ ∈ l.W + 1 : n])
    append!(l.morpheme_symbols, [I[] for _ ∈ l.W + 1 : n])
    append!(l.sequences, [Set(Vector{I}[]) for _ ∈ l.W + 1 : n])
    append!(l.next_layer_symbol, [false for _ ∈ l.W + 1 : n])

    l.W = n
    l.stop_growth = false

    return nothing
end