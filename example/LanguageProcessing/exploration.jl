

"""alphabet classification"""
function explore_text!(a::Adam, text::String, threshold::Int,
    substitute::String = "unk", logs = [])::String

    t_0 = Dates.now()
    msg_to_logs(string("text exploration started: ",
    Dates.format(t_0, "yyyy-mm-dd HH:MM:SS"), "\n"), logs)

    l = a.layers[1]
    h = l.semantic[1]

    l.parser.memorise = false

    words, popular_words = find_words(text, threshold)
    words_sum = mapreduce(x -> x[2], +, collect(words))
    popular_words_sum = mapreduce(x -> x[2], +, collect(popular_words))

    to_change = join([
        "to change: ",
        @sprintf("%.2f", 100.0 * (words_sum - popular_words_sum) / words_sum),
        "% words, ",
        @sprintf("%.2f", 100.0 * (length(words) - length(popular_words)) / length(words)),
        "% unique words, "])
    msg_to_logs(string(to_change, '\n'), logs)

    words_to_parse = filter(x -> x != "-", unique(map(
        word -> join([lowercase(word), " "]), collect(keys(popular_words)))))
    if substitute != ""
        push!(words_to_parse, join([substitute, " "]))
    end

    #=println("words for exploration: ", '\n')
    for word ∈ words_to_parse
        println(word)
    end=#

    targets = map(word -> map(x -> isempty(x) ? M(0) : x[1], encode(word, a)), words_to_parse)

    i = 0
    while !isempty(words_to_parse) && h.D < h.M && i < 100

        i += 1

        #println(h.D)
        msg_to_logs(string("iteration: $i, words to parse: ",
        length(words_to_parse), '\n'), logs)
        for word in words_to_parse[1:min(3, length(words_to_parse))]
            println(word)
        end

        text_to_parse = join(words_to_parse)
        parse_text!(a, text_to_parse, false, 1)

        not_learned = map(x -> !haskey(h.string_idx, x), targets)

        targets = targets[not_learned]
        words_to_parse = words_to_parse[not_learned]
    end

    unlearned_words = Set(map(word -> strip(word, ' '), words_to_parse))

    substitute = "|"

    word_pattern = r"\b([a-z]|[A-Z]|[А-Я]|[а-я]|Ё|ё|\d|-)+\b"
    change(word) = !haskey(popular_words, lowercase(word)) ? substitute :
        word ∈ unlearned_words ? substitute : word
    cleaned_text = replace(text, word_pattern => x -> change(x))
    #cleaned_text = remove_extra_spaces(cleaned_text)
    cleaned_text = replace(cleaned_text, "| " => "|")

    l.parser.memorise = l.parser.memory_length > 0

    t_1 = Dates.now()
    msg_to_logs(string("text exploration completed: ",
    Dates.format(t_1, "yyyy-mm-dd HH:MM:SS"),
    " ($(time_interval(t_0, t_1)))\n\n"), logs)

    return cleaned_text
end

function simplify_text!(text::String, threshold::Int, logs = [])::String

    println("start simplify_text!")
    t_0 = Dates.now()
    msg_to_logs(string("text simplification started: ",
    Dates.format(t_0, "yyyy-mm-dd HH:MM:SS"), "\n"), logs)

    words_v = lowercase.(split(text, " "))
    words::Dict{String, I} = countmap(words_v)
    popular_words = filter(x -> x[2] > threshold, words)

    words_sum = mapreduce(x -> F(x[2]), +, collect(words))
    popular_words_sum = mapreduce(x -> F(x[2]), +, collect(popular_words))

    to_change = join([
        "to change: ",
        @sprintf("%.2f", 100.0 * (words_sum - popular_words_sum) / words_sum),
        "% words, ",
        @sprintf("%.2f", 100.0 * (length(words) - length(popular_words)) / length(words)),
        "% unique words, "])
    msg_to_logs(string(to_change, '\n'), logs)

    substitute = "|"

    change(word) = !haskey(popular_words, word) ? substitute :
        word * " "
    
    words_v = map(word -> change(word), words_v)

    simplified_text = join(words_v)

    t_1 = Dates.now()
    msg_to_logs(string("text simplification completed: ",
    Dates.format(t_1, "yyyy-mm-dd HH:MM:SS"),
    " ($(time_interval(t_0, t_1)))\n\n"), logs)

    return simplified_text
end


 