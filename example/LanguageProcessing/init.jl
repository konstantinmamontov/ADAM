

"""alphabet classification"""
function make_alphabet!(a::Adam)::Nothing
    
    a.app["small letters en"] = Char.('a':'z')
    a.app["small letters rus"] = [Char.('а':'я'); ['ё']]
    a.app["capital letters en"] = Char.('A':'Z')
    a.app["capital letters rus"] = [Char.('А':'Я'); ['Ё']]

    a.app["digits"] = Char.('0':'9')
    
    a.app["apostrophes"] = ['\'']

    a.app["dashes"] = ['-', Char(8211), Char(8212), Char(8213), Char(8722), Char(9472)] # different dashes

    a.app["separating punctuation"] = [',', ';', ':']
    a.app["ending punctuation"] = ['.', '!', '?']

    a.app["punctuation"] = ['.', '!', '?', ',', ';', ':', '%', '$', Char(8230)]

    #=a.app["separating punctuation"] = [',', ';', '(', ')', '[', ']', ':']
    a.app["ending punctuation"] = ['.', '!', '?']

    a.app["punctuation"] = ['.', '!', '?', ',', ';', '(', ')',
    '[', ']', ':', '%', '/', '$', Char(8230)]=#
    
    a.app["whitespaces"] = [' ', '\u00A0', '\t', '\n', '_', Char(8194), Char(8197),
    Char(8200), Char(8201), Char(8239)]

    return nothing
end

"""many -> one encoding addition"""
function connect!(coder::Dict, decoder::Dict,
    symbols::Vector{Char}, code_symbol::Char)::Nothing

    for symbol in symbols
        coder[symbol] = code_symbol
    end
    decoder[code_symbol] = symbols

    return nothing
end

"""one -> oneself encodings addition"""
function identity!(coder::Dict, decoder::Dict,
    symbols::Vector{Char})::Nothing

    for symbol in symbols
        coder[symbol] = symbol
        decoder[symbol] = [symbol]
    end

    return nothing
end

"""encoding/decoding dict making"""
function make_coder!(a::Adam)::Nothing

    l = a.layers[1]

    a.app["encoder"] = Dict(
        h.N => Dict{Char, Char}() for h in l.semantic)
    a.app["decoder"] = Dict(
        h.N => Dict{Char, Vector{Char}}() for h in l.semantic)

    # head №1 char -> char encoding

    encoder_1 = a.app["encoder"][1]
    decoder_1 = a.app["decoder"][1]

    for letter in [a.app["small letters en"]; a.app["small letters rus"]]
        lower = letter
        upper = uppercase(letter)
        encoder_1[lower] = lower
        encoder_1[upper] = lower
        decoder_1[lower] = [lower, upper]
    end
    
    identity!(encoder_1, decoder_1,
    [a.app["digits"]; a.app["punctuation"]; a.app["dashes"]])

    connect!(encoder_1, decoder_1, a.app["whitespaces"], '_')
    connect!(encoder_1, decoder_1, a.app["apostrophes"], '\'')
    #encoder_1['_'] = '_'
    #decoder_1['_'] = [decoder_1['_']; ['_']]
    
    #encoder_1[' '] = '_'
    #decoder_1['_'] = [' ']

    #encoder_1['\n'] = '|'
    #decoder_1['|'] = ['\n']

    if a.N[1] >= 2

        # head №2 char -> char encoding

        encoder_2 = a.app["encoder"][2]
        decoder_2 = a.app["decoder"][2]

        # russian
        connect!(encoder_2, decoder_2, a.app["small letters rus"], 'а')
        connect!(encoder_2, decoder_2, a.app["capital letters rus"], 'А')

        # english
        connect!(encoder_2, decoder_2, a.app["small letters en"], 'a')
        connect!(encoder_2, decoder_2, a.app["capital letters en"], 'A')

        # digits
        connect!(encoder_2, decoder_2, a.app["digits"], '0')

        # punctuation, dashes
        identity!(encoder_2, decoder_2,
        [a.app["punctuation"]; a.app["dashes"]])

        # others

        code_char = '_'

        decoder_2[code_char] = Vector{Char}()

        for symbol in keys(encoder_1)
            if !haskey(encoder_2, symbol)
                encoder_2[symbol] = code_char
                push!(decoder_2[code_char], symbol)
            end
        end

    end

    # layer №1 encoding 

    a.app["char code -> code"] = Dict(
        h.N => Dict(char_code => M(i) for (i,(char_code, initial_chars)) in 
        enumerate(a.app["decoder"][h.N])) for h in l.semantic)

    a.app["code -> char code"] = Dict(
        h.N => Dict(code => char_code for (char_code, code) in 
        a.app["char code -> code"][h.N]) for h in l.semantic)

    a.app["char -> h code"] = Dict(
        h.N => Dict(initial_char =>
        a.app["char code -> code"][h.N][a.app["encoder"][h.N][initial_char]]
        for initial_char in keys(a.app["encoder"][h.N]))
            for h in l.semantic)

    a.app["char -> l code"] = Dict(
        initial_char => M[a.app["char -> h code"][h.N][initial_char] for 
        h in l.semantic] for initial_char in keys(a.app["encoder"][1]))

    return nothing
end

"""Adam initialization with application specifics"""
function initialize(p::Dict=Dict(), logs = [])::Adam

    t_0 = Dates.now()
    msg_to_logs(string("ADAM initialization started: ",
    Dates.format(t_0, "yyyy-mm-dd HH:MM:SS"), "\n"), logs)

    application_parameters = get(p, "application_parameters", Dict())
    a = new_adam(p["ADAM_parameters"])
    a.app = deepcopy(application_parameters)

    make_alphabet!(a)
    make_coder!(a)

    l = a.layers[1]

    # structures parameters, heads alphabet

    l.K = [length(a.app["decoder"][i]) for i in 1:l.N]
    a.d[1] = 1
    l.d = 1

    for h in l.semantic
        h.d = 1 # DEL 0.6.0
        h.k = l.K[h.N]
        h.D = h.k
        h.string = [[[x] for x=1:h.k]; [[] for _ in h.k+1:l.M]]
        h.string_idx = Dict([x] => x for x ∈ 1 : h.k)
        h.v = [zeros(5) for _ in 1:l.M]
    end

    # v.0.5.0+ layer morphemes creation
    map(w -> encode!(l, M[w]), 1:l.K[1])

    t_1 = Dates.now()
    msg_to_logs(string("ADAM initialization completed: ",
    Dates.format(t_1, "yyyy-mm-dd HH:MM:SS"),
    " ($(time_interval(t_0, t_1)))\n\n"), logs)

    return a
end

function update!(a::Adam, parameters::Dict)::Nothing

    if haskey(parameters, "decode_slow")
        decode_slow = parameters["decode_slow"]
        a.decode_slow = decode_slow
        for l ∈ a.layers
            l.planner.decode_slow = (
                l.L <= length(decode_slow) ? decode_slow[l.L] : decode_slow[end])
        end
    end

    if haskey(parameters, "decode_num")
        decode_num = parameters["decode_num"]
        a.decode_num = decode_num
        for l ∈ a.layers
            l.planner.decode_num = (
                l.L <= length(decode_num) ? decode_num[l.L] : decode_num[end])
        end
    end

    if haskey(parameters, "planner_order")
        planner_order = parameters["planner_order"]
        a.planner_order = planner_order 
        for l ∈ a.layers
            l.planner.order = (
                l.L <= length(planner_order) ? planner_order[l.L] : planner_order[end])
        end
    end

    if haskey(parameters, "plans_min")
        plans_min = parameters["plans_min"]
        a.plans_min = plans_min 
        for l ∈ a.layers
            l.planner.min = (
                l.L <= length(plans_min) ? plans_min[l.L] : plans_min[end])
        end
    end

    if haskey(parameters, "planning_fast")
        planning_fast = parameters["planning_fast"]
        a.planning_fast = planning_fast
        for l ∈ a.layers
            l.planner.planning_fast = (
                l.L <= length(planning_fast) ? planning_fast[l.L] : planning_fast[end])
        end
    end

    if haskey(parameters, "planning_slow")
        planning_slow = parameters["planning_slow"]
        a.planning_slow = planning_slow
        for l ∈ a.layers
            l.planner.planning_slow = (
                l.L <= length(planning_slow) ? planning_slow[l.L] : planning_slow[end])
        end
    end

    if haskey(parameters, "quorum")
        quorum = parameters["quorum"]
        a.quorum = quorum
        for l ∈ a.layers
            l.planner.quorum = (
                l.L <= length(quorum) ? quorum[l.L] : quorum[end])
        end
    end

    if haskey(parameters, "fitness")
        fitness = parameters["fitness"]
        a.fitness = fitness
        for l ∈ a.layers
            l.fitness = (
                l.L <= length(fitness) ? fitness[l.L] : fitness[end])
        end
    end

    return nothing
end

function remove_unwanted_symbols(s::String, allowed_set::Set{Char})
    buffer = IOBuffer()
    for c in s
        if c in allowed_set
            write(buffer, c)
        else
            write(buffer, ' ')
        end
    end
    return String(take!(buffer))
end

function surround_punctuation_with_spaces(s::String, allowed_set::Set{Char})
    buffer = IOBuffer()
    prev_char = '\0'
    prev_allowed_char = '\0'

    for c in s
        if c in allowed_set
            if c == prev_allowed_char
                continue
            end
            if prev_char != ' '
                write(buffer, ' ')
            end
            write(buffer, c)
            write(buffer, ' ')
            prev_allowed_char = c
        else
            write(buffer, c)
            if c != ' '
                prev_allowed_char = '\0'
            end
        end
        prev_char = c
    end

    return String(take!(buffer))
end

function surround_dashes_with_spaces(s::String, allowed_set::Set{Char})
    buffer = IOBuffer()
    prev_char = '\0'

    for c in s
        if c in allowed_set
            write(buffer, '-')
            if prev_char == ' '            
                write(buffer, ' ')
                prev_char = ' '
                continue
            end
        else
            write(buffer, c)
        end
        prev_char = c
    end

    return String(take!(buffer))
end

function whitespaces_conversion(s::String, allowed_set::Set{Char})
    buffer = IOBuffer()
    for c in s
        if c in allowed_set
            write(buffer, ' ')
        else
            write(buffer, c)
        end
    end

    return String(take!(buffer))
end

function remove_extra_spaces(s::String)
    buffer = IOBuffer()
    prev_char = '\0'

    for c in s
        if c == ' ' && prev_char == ' '
            continue
        end
        write(buffer, c)
        prev_char = c
    end

    return String(take!(buffer))
end

function remove_external_apostrophes(s::String, apostrophes_set::Set{Char}, letters_set::Set{Char})
    buffer = IOBuffer()
    prev_char = '\0'
    prev_prev_char = '\0'

    for c in s
        if prev_prev_char in letters_set && prev_char in apostrophes_set && c in letters_set
            write(buffer, '\'')
            write(buffer, c)
        elseif !(c in apostrophes_set)
            write(buffer, c)
        end
        prev_prev_char = prev_char
        prev_char = c
    end

    return String(take!(buffer))
end

function clean_text!(text::String, a::Adam)::String
    
    #println("Выполнение функции")
    all_symbols_set = Set(keys(a.app["char -> l code"]))
    punctuation_set = Set(a.app["punctuation"])
    whitespaces_set = Set(a.app["whitespaces"])
    dashes_set = Set(a.app["dashes"])
    #apostrophes_set = Set(a.app["apostrophes"])
    apostrophes_set = Set(get(a.app, "apostrophes", Char[]))
    eng_letters = Set([a.app["capital letters en"]; a.app["small letters en"]])

    text = remove_unwanted_symbols(text, all_symbols_set)
    #println("Текст почищен от неизвестных символов!")
    
    text = whitespaces_conversion(text, whitespaces_set)
    #println("Текст поменял пробелы!")

    text = surround_punctuation_with_spaces(text, punctuation_set)
    #println("Текст окружен пробелами!")

    text = surround_dashes_with_spaces(text, dashes_set)
    #println("Текст обработал тире!")

    text = remove_extra_spaces(text)
    #println("Текст потерял пробелы!")

    text = remove_external_apostrophes(text, apostrophes_set, eng_letters)

    return text
end

function clean_text2!(input_file::String, a::Adam)::String
    t0=time()
    output_file = split(input_file, ".")[1] * "_cleaned.txt"
    #println("Выполнение функции")
    all_symbols_set = Set(keys(a.app["char -> l code"]))
    punctuation_set = Set(a.app["punctuation"])
    whitespaces_set = Set(a.app["whitespaces"])
    dashes_set = Set(a.app["dashes"])
    #apostrophes_set = Set(a.app["apostrophes"])
    apostrophes_set = Set(get(a.app, "apostrophes", Char[]))
    eng_letters = Set([a.app["capital letters en"]; a.app["small letters en"]])

    io = IOBuffer()
    fid = open(input_file); chunk = 1_000_000
    lines_src = countlines(input_file); lines_in =0
    
    while !eof(fid)
        line = readline(fid);   lines_in +=1;
        
        line = remove_unwanted_symbols(line, all_symbols_set)
        #println("Текст почищен от неизвестных символов!")
        
        line = whitespaces_conversion(line, whitespaces_set)
        #println("Текст поменял пробелы!")

        line = surround_punctuation_with_spaces(line, punctuation_set)
        #println("Текст окружен пробелами!")

        line = surround_dashes_with_spaces(line, dashes_set)
        #println("Текст обработал тире!")

        line = remove_extra_spaces(line)
        #println("Текст потерял пробелы!")

        line = remove_external_apostrophes(line, apostrophes_set, eng_letters)
    
        print(io, line * '\n')

        if lines_in%chunk == 0
            t_ = chunk * ((time() - t0) / lines_in)
            t_ = round(t_, digits=3)
            done = round(100*(lines_in / lines_src), digits=2)
            println(done,"%  lines_in=", lines_in, "  t_chunk=", t_)
            flush(stdout)
        end

    end

    text2 = String(take!(io))
    write(output_file, text2)
    close(fid)
    println("lines in=", lines_in)
    println(output_file, " saved")
    t1=time()
    println("total ", round((t1-t0)/3600, digits=3)," hours")

    return output_file
end