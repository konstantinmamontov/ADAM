

function parse_empty!(l::Layer, need_episodic::Bool=false)

    (w_, r_, t_) = parse_input!(l, I(0), M[], R(0.0), T(0))

    need_episodic && fill_episodic!(l.episodic, I(0), M[], R(0.0), T(1))

    return (w_, r_, t_)
end

function parse_symbol!(l::Layer, s::Morpheme, t::T=T(0),
    need_episodic::Bool=false)

    (i, m, w) = encode!(l, s)
    (w_, r_, t_) = parse_input!(l, i, w, R(0.0), t)

    need_episodic && fill_episodic!(l.episodic, i, w, R(0.0), t)

    return (w_, r_, t_)
end

parse_and_separate!(l::Layer, s::Morpheme, t::T=T(0),
    need_episodic::Bool=false) = (
        parse_symbol!(l, s, t, need_episodic);
        parse_empty!(l, need_episodic)
    )

parse_separated_symbol!(l::Layer, s::Morpheme, t::T=T(0),
    need_episodic::Bool=false) = (
        parse_empty!(l, need_episodic);
        parse_and_separate!(l, s, t, need_episodic)
    )

parse_whitespace!(l::Layer, s::Morpheme, t::T=T(0), need_episodic::Bool=false) =
    (isempty(l.parser.stack[end-1][1]) || 
    ([l.semantic[1].string[l.parser.stack[end-1][1][1]][end]] != s)) && (
        parse_and_separate!(l, s, t, need_episodic))

parse_dash!(l::Layer, s::Morpheme, t::T=T(0), need_episodic::Bool=false) =
    isempty(l.parser.stack[end][1]) ? 
    parse_and_separate!(l, s, t, need_episodic) :
    parse_symbol!(l, s, t, need_episodic)

parse_punctuation!(l::Layer, s::Morpheme, t::T=T(0),
    need_episodic::Bool=false) =

    #-isempty(l.parser.stack[end][1]) ?
    #-parse_and_separate!(l, s, t, need_episodic) :
    #-parse_separated_symbol!(l, s, t, need_episodic)
    
    #-parse_and_separate!(l, s, t, need_episodic)
    parse_symbol!(l, s, t, need_episodic)

function parse_char!(a::Adam, char::Char, t::T=T(0), need_episodic::Bool=false)::Nothing
    if char == '|'
        return nothing
    end
    s = encode(char, a)
    l = a.layers[1]

    if char == ' '
        #parse_whitespace!(l, s, t, need_episodic)
        parse_and_separate!(l, s, t, need_episodic)
    #elseif char ∈ a.app["dashes"]
    #    parse_dash!(l, s, t, need_episodic)
    #elseif char ∈ a.app["punctuation"]
    #    word_end = encode(' ', a)
    #    parse_whitespace!(l, word_end, t, need_episodic)
    #    parse_punctuation!(l, s, t, need_episodic)
    #    parse_whitespace!(l, word_end, t, need_episodic)
    #elseif !isempty(s)
    #    parse_symbol!(l, s, t, need_episodic)
    #elseif !isempty(l.parser.stack[end][1])
    #    parse_empty!(l, need_episodic)
    else
        #whitespace = encode(' ', a)
        #parse_whitespace!(l, whitespace, t, need_episodic)
        parse_symbol!(l, s, t, need_episodic)
    end
    return nothing
end

parse_text!(a::Adam, text::Vector{String}, need_episodic::Bool=false,
    iterations::Int64=1)::Nothing = parse_text!(
        a, join(text), need_episodic, iterations)

function parse_text!(a::Adam, text::String, need_episodic::Bool=false,
    iterations::Int64=1, logs = [])::Nothing

    t_0 = Dates.now()
    msg_to_logs(string("training layer #$(a.L) started: ",
    Dates.format(t_0, "yyyy-mm-dd HH:MM:SS"), "\n"), logs)

    t = T(1)

    if need_episodic
        e = a.layers[1].episodic
        max_length = Int(round(length(text) * iterations * 1.4))
        pre_allocate!(e, max_length)
    end

    last_time = Dates.now()
    chunk = 10_000_000
    counter = 0
    for input_symbol ∈ text
        parse_char!(a, input_symbol, t, need_episodic)
        counter += 1
        if counter % chunk == 0
            new_time = Dates.now()
            msg_to_logs(string("$(counter) symbols parsed: ",
            Dates.format(new_time, "yyyy-mm-dd HH:MM:SS"),
            " (last chunk = $(chunk) for $(time_interval(last_time, new_time)))\n"), logs)
            last_time = new_time
        end
    end

    for _ ∈ 2:iterations
        if need_episodic
            training_reparse!(a.layers[1])
        else
            for input_symbol ∈ text
                parse_char!(a, input_symbol, t, need_episodic)
            end
        end
    end

    t_1 = Dates.now()
    msg_to_logs(string("training layer #$(a.L) completed: ",
    Dates.format(t_1, "yyyy-mm-dd HH:MM:SS"),
    " ($(time_interval(t_0, t_1)))\n"), logs)

    return nothing
end

reparse_and_separate!(l::Layer, s::Morpheme,
    t::T=T(0))::Vector{Tuple{Morpheme, R, T}} = [
        parse_symbol!(l, s, t), parse_empty!(l)]

reparse_separated_symbol!(l::Layer, s::Morpheme,
    t::T=T(0))::Vector{Tuple{Morpheme, R, T}} = [
        [parse_empty!(l)]; reparse_and_separate!(l, s, t)]

reparse_whitespace!(l::Layer, s::Morpheme,
    t::T=T(0))::Vector{Tuple{Morpheme, R, T}} = 
    (isempty(l.parser.stack[end-1][1]) || 
    ([l.semantic[1].string[l.parser.stack[end-1][1][1]][end]] != s)) ?
    reparse_and_separate!(l, s, t) : Tuple{Morpheme, R, T}[]

reparse_dash!(l::Layer, s::Morpheme,
    t::T=T(0))::Vector{Tuple{Morpheme, R, T}} =
        isempty(l.parser.stack[end][1]) ? reparse_and_separate!(l, s, t) :
        [parse_symbol!(l, s, t)]

reparse_punctuation!(l::Layer, s::Morpheme,
    t::T=T(0))::Vector{Tuple{Morpheme, R, T}} = 
        #isempty(l.parser.stack[end][1]) ? reparse_and_separate!(l, s, t) :
        #reparse_separated_symbol!(l, s, t)
        [parse_symbol!(l, s, t)]

function reparse_char!(a::Adam, char::Char,
    t::T=T(0))::Vector{Tuple{Morpheme, R, T}}
    
    s = encode(char, a)
    l = a.layers[1]

    if char ∈ a.app["whitespaces"]
        return reparse_whitespace!(l, s, t)
    elseif char ∈ a.app["dashes"]
        return reparse_dash!(l, s, t)
    elseif char ∈ a.app["punctuation"] 
        word_end = encode(' ', a)
        return vcat(
            [reparse_whitespace!(l, word_end, t),
            reparse_punctuation!(l, s, t),
            reparse_whitespace!(l, word_end, t)]...)
    elseif !isempty(s) 
        return [parse_symbol!(l, s, t)]
    #elseif !isempty(l.parser.stack[end][1])
    #    return [parse_empty!(l)]
    else
        whitespace = encode(' ', a)
        return reparse_whitespace!(l, whitespace, t)         
    end
    return Tuple{Morpheme, R, T}[]
end

"""Layer №1 reparsing based on text""" 
function reparse_text!(a::Adam,
    text::String, logs=[])::Tuple{Vector{Morpheme}, Vector{R}, Vector{T}}

    t_0 = Dates.now()
    msg_to_logs(string("reparsing layer #1 started: ",
    Dates.format(t_0, "yyyy-mm-dd HH:MM:SS"), "\n"), logs)

    l = a.layers[1]
    clear_parser!(l)

    wrtv = Tuple{Morpheme, R, T}[]
    t = 1

    size = div(length(text), 4)
    sizehint!(wrtv, size)

    for char in text

        new_output = reparse_char!(a, char, t)
        
        if !isempty(new_output)
            filter!(wrt -> !isempty(wrt[1]), new_output)
            append!(wrtv, new_output)
            t += 1
        end

    end

    wv = map(wrt -> wrt[1], wrtv)
    rv = map(wrt -> wrt[2], wrtv)
    tv = map(wrt -> wrt[3], wrtv)

    (W_, R_, T_) = clear_parser!(l)
    append!(wv, W_)
    append!(rv, R_)
    append!(tv, T_)

    t_1 = Dates.now()
    msg_to_logs(string("reparsing layer #1 completed: ",
    Dates.format(t_1, "yyyy-mm-dd HH:MM:SS"),
    " ($(time_interval(t_0, t_1)))\n\n"), logs)

    return (wv, rv, tv)
end

fill_episodic!(e::Episodic, i::I, w::Morpheme, r::R, t::T) = (
    push!(e.i, i);
    push!(e.w, w);
    push!(e.r, r);
    push!(e.t, t);
    e.length += 1
)

function pre_allocate!(e::Episodic, n::Int)::Nothing

    sizehint!(e.i, n)
    sizehint!(e.w, n)
    sizehint!(e.r, n)
    sizehint!(e.t, n)

    return nothing
end

function training_reparse!(l::Layer)::Nothing

    e = l.episodic

    for t ∈ 1 : e.length
        parse_input!(l, e.i[t], e.w[t], e.r[t], e.t[t])
    end

    return nothing
end

