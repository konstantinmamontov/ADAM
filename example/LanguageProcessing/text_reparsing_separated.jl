

function text_reparsing_history!(a::Adam, text::String, logs = [])::Vector{I}

    t_start = Dates.now()
    msg_to_logs(string("reparsing layer #1 started: ",
    Dates.format(t_start, "yyyy-mm-dd HH:MM:SS"), "\n"), logs)

    l = a.layers[1]
    #p = l.parser
    #h = l.semantic[1]
    #stack = p.stack
    clear_parser!(l)

    m_history = I[]
    current_w = M[]

    estimated_size = div(length(text), 4)
    sizehint!(m_history, estimated_size)

    t_0 = T(1)
    r = R(0.0)
    t_strong_sep = T(-100)
    t_sep_punct = T(-2)
    t_end_punct = T(-3)

    t = t_0

    t_ = t_0
    r_ = R(0.0)

    whitespaces = Set(a.app["whitespaces"])
    #punctuation = Set(a.app["punctuation"])
    #dashes = Set(a.app["dashes"])
    separating_punctuation = Set(a.app["separating punctuation"])
    ending_punctuation = Set(a.app["ending punctuation"])

    #whitespace_code = encode(' ', a)

    prev_char = '\0'

    last_time = Dates.now()
    chunk = 10_000_000
    counter = 0
    for char ∈ text

        if char == '|'
            if prev_char == '|'
                continue
            end
            current_w, r_, t_ = parse_input!(l, I(0), M[], r, t_strong_sep)
            if !isempty(current_w)
                if t_ != t_0
                    #=if t_ < T(-2)
                        push!(m_history, I(t_))
                    end=#
                    push!(m_history, l.index[current_w])
                    push!(m_history, I(t_))
                else
                    push!(m_history, l.index[current_w])
                end
            elseif t_ == t_strong_sep
                push!(m_history, I(t_))
            end
            prev_char = char
            continue
        end

        if prev_char ∈ separating_punctuation
            t = t_sep_punct
        elseif prev_char ∈ ending_punctuation 
            t = t_end_punct
        else
            t = t_0
        end

        s = encode(char, a)

        #=if isempty(s) || char ∈ whitespaces

            if (isempty(stack[end-1][1]) || 
                (h.string[stack[end-1][1][1]][end] != whitespace_code[1])) #previous symbol is not whitespace

                #whitespase parsing

                (i, m, w) = encode!(l, s)
                current_w = parse_input!(l, i, w, r, t)[1]
                if !isempty(current_w)
                    push!(w_history, current_w[1])
                end

                #empty morpheme parsing

                current_w = parse_input!(l, I(0), M[], r, T(0))[1]
                if !isempty(current_w)
                    push!(w_history, current_w[1])
                end
            end

        elseif char ∈ dashes

            (i, m, w) = encode!(l, s)

            need_separation = isempty(stack[end][1]) #previous symbol is whitespace

            current_w = parse_input!(l, i, w, r, t)[1]
            if !isempty(current_w)
                push!(w_history, current_w[1])
            end

            if need_separation

                #next symbol is forced whitespace

                current_w = parse_input!(l, I(0), M[], r, T(0))[1]
                if !isempty(current_w)
                    push!(w_history, current_w[1])
                end

            end
        
        elseif char ∈ punctuation

            #previous symbol is forced whitespace

            if (isempty(stack[end-1][1]) || 
                (h.string[stack[end-1][1][1]][end] != whitespace_code[1]))

                (i, m, w) = encode!(l, whitespace_code)
                current_w = parse_input!(l, i, w, r, t)[1]
                if !isempty(current_w)
                    push!(w_history, current_w[1])
                end

                current_w = parse_input!(l, I(0), M[], r, T(0))[1]
                if !isempty(current_w)
                    push!(w_history, current_w[1])
                end
            end

            
            (i, m, w) = encode!(l, s)
            current_w = parse_input!(l, i, w, r, t)[1]
            if !isempty(current_w)
                push!(w_history, current_w[1])
            end

            #next symbol is forced whitespace

            (i, m, w) = encode!(l, whitespace_code)
            current_w = parse_input!(l, i, w, r, t)[1]
            if !isempty(current_w)
                push!(w_history, current_w[1])
            end

            current_w = parse_input!(l, I(0), M[], r, T(0))[1]
            if !isempty(current_w)
                push!(w_history, current_w[1])
            end

        else # common char

            (i, m, w) = encode!(l, s)
            current_w = parse_input!(l, i, w, r, t)[1]
            if !isempty(current_w)
                push!(w_history, current_w[1])
            end

        end =#

        (i, m, w) = encode!(l, s)
        current_w, r_, t_ = parse_input!(l, i, w, r, t)
        if !isempty(current_w)
            if t_ != t_0
                #=if t_ < T(-2)
                    push!(m_history, I(t_))
                end=#
                push!(m_history, l.index[current_w])
                push!(m_history, I(t_))
            else
                push!(m_history, l.index[current_w])
            end
        elseif t_ == t_strong_sep
            push!(m_history, I(t_))
        end

        if char ∈ whitespaces

            #empty morpheme parsing

            current_w, r_, t_ = parse_input!(l, I(0), M[], r, t_0)
            if !isempty(current_w)
                if t_ != t_0
                    #=if t_ < T(-2)
                        push!(m_history, I(t_))
                    end=#
                    push!(m_history, l.index[current_w])
                    push!(m_history, I(t_))
                else
                    push!(m_history, l.index[current_w])
                end
            elseif t_ == t_strong_sep
                push!(m_history, I(t_))
            end

        end

        prev_char = char

        counter += 1
        if counter % chunk == 0
            new_time = Dates.now()
            msg_to_logs(string("$(counter) symbols reparsed: ",
            Dates.format(new_time, "yyyy-mm-dd HH:MM:SS"),
            " (last chunk = $(chunk) for $(time_interval(last_time, new_time)))\n"), logs)
            last_time = new_time
        end

    end

    W_, R_, T_= clear_parser!(l)
    for (i, w) ∈ enumerate(W_)
        t_ = T_[i]
        if t_ == t_strong_sep
            push!(m_history, I(t_))
        elseif t_ == t_0
            push!(m_history, l.index[w])
        else
            #push!(m_history, I(t_))
            push!(m_history, l.index[w])
            push!(m_history, I(t_))
        end
    end

    t_end = Dates.now()
    msg_to_logs(string("reparsing layer #1 completed: ",
    Dates.format(t_end, "yyyy-mm-dd HH:MM:SS"),
    " ($(time_interval(t_start, t_end)))\n\n"), logs)

    return m_history
end