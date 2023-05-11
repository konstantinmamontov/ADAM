

""" Print message to ostreams """
msg_to_logs(msg::String, logs=[]) =
    map(log -> log_msg(log, msg), logs)

function log_msg(ostream, msg::String)::Nothing

    print(ostream, msg)
    flush(ostream)

    return nothing
end

""" Pretty size output """
function pretty_size(path::String)::String

    units = ["b", "Kb", "Mb", "Gb", "Tb"]
    size = Float(filesize(path))
    i = 1
    while size > 1000.0 && i < length(units)
        size /= 1024.0
        i += 1
    end

    return @sprintf("%.2f %s", size, units[i])
end



# layer №1 parsing analytics output

function show_morphemes(a::Adam, n::Int=1, ostream=Base.stdout)
    l = a.layers[1]       
    h = l.semantic[n] 

    alphabet = sort(
        map(w -> decode(w, a, n), filter(x -> length(x) == 1, h.string)))

    print(ostream, "Initial alphabet:\n")
    print(ostream, alphabet[1])
    char_code = Int(alphabet[1][1])
    for s in alphabet[2:end]
        new_char_code = Int(s[1])
        delimiter = abs(new_char_code - char_code) > 1 ? '\n' : ' '
        print(ostream, delimiter, s)
        char_code = new_char_code
    end

    separations = a.app["1_st_layer_parsing"]["morphemes_separations"]

    println(ostream, "\n\nFormed morphemes:")

    for (w, ww) in separations
        println(ostream, separate_morpheme(a, w, separations))
    end
end

learned_word_representation(a::Adam, word::String, counter::Int)::String =
    "$word ($(a.app["1_st_layer_parsing"]["words_separations"][word]))" *
    " - $counter times\n"


function show_learned_words(a::Adam, ostream=Base.stdout)::Nothing
    print(ostream,
        "Learned words and their occurrence in the text:\n")

    output_data = sort(collect(
        a.app["1_st_layer_parsing"]["words learned"]), by=x->x[2], rev=true)

    for (word, counter) ∈ output_data
        print(ostream, learned_word_representation(a, word, counter))
    end
end

function show_learned_popular_words(a::Adam, ostream=Base.stdout)::Nothing
    print(ostream,
        "Learned popular words and their occurrence in the text:\n")

    output_data = sort(collect(
        a.app["1_st_layer_parsing"]["words learned (popular)"]), by=x->x[1])

    for (word, counter) in output_data
        print(ostream, learned_word_representation(a, word, counter))
    end
end

function print_stats(d::Dict, ostream = Base.stdout)::Nothing
    
    pairs = map(p -> (string(p[1]), string(p[2])), sort(collect(d), by = x -> x[1]))

    tab = maximum(p -> length(p[1]), pairs)

    for (key, value) in pairs
        msg_to_logs(string((key * ' '^tab)[1:tab], " => ", value, "\n"), [ostream])
    end

end