using InteractiveUtils
function get_mem_use()
    f = open( "/proc/self/stat" )
    s = read( f, String )
    vsize = parse( Int64, split( s )[23] )
    mb = Int( ceil( vsize / ( 1024 * 1024 ) ) )
    gb = Int( ceil( vsize / ( 1024 * 1024 * 1024 ) ) )
    return gb
end

function run_learning(app_path::String)::Nothing

    start_time = Dates.now()

    parameters = load_json(joinpath(app_path, "Learning_full_parameters.json"))
    test_parameters = parameters["test_parameters"]

    start = Dates.format(start_time, "yyyy_mm_dd: H_M_S")

    today = split(start, ':')[1]
    time = split(start, ' ')[2]
    title = parameters["Test"] * "_" * time

    text_file = joinpath(app_path, "articles",
        parameters["test_parameters"]["text_file"])

gb = get_mem_use()
println("before read text ", gb)
        
    text = read(text_file, String)

    text_repetition = get(test_parameters, "text_repetition", 1)

    report_path = mkpath(joinpath(app_path, "reports", today, title))

    log_file = open(joinpath(report_path, "log.txt"), "w")
    logs = [Base.stdout, log_file]

    msg_to_logs(string("program execution started: ",
    Dates.format(start_time, "yyyy-mm-dd HH:MM:SS"), "\n\n"), logs)

    msg_to_logs(string(
        "text: $text_file ($(pretty_size(text_file)))", "\n\n"), logs)

    a = initialize(parameters, logs)

    threshold = get(test_parameters, "unknown_threshold", 0)

#run(`ls -l /proc/self`)
#gb = get_mem_use()
#println("/proc/self/stat in  GB=", gb)

#println("Base.gc_live_bytes() in GB =", Base.gc_live_bytes()/10^9)

    #text = clean_text!(text, a)
    text = clean_text2!(text_file, a)
    
    exit()
println("size of text=", Base.summarysize(text))
println("varinfo=",varinfo())
gb = get_mem_use()
println("GB=", gb)
println("Base.gc_live_bytes() in GB =", Base.gc_live_bytes()/10^9)
println("before GC=", gb)
GC.gc()
println("after GC=", gb)




gb = get_mem_use()
println("before simplify! ", gb)

    text = simplify_text!(text, threshold, logs)

gb = get_mem_use()
println("GB=", gb)

println("after simplify! ", gb)

println("Base.gc_live_bytes()in GB =", Base.gc_live_bytes()/10^9)
exit()

    parse_text!(a, text, false, text_repetition, logs)

    report_file = open(joinpath(report_path, "report.txt"), "w")

    println(report_file, "Time: ", string(Dates.format(start_time, "yyyy-mm-dd HH:MM:SS")))
    println(report_file, "Text: $text_file ($(pretty_size(text_file)))")
    println(report_file, "Parameters:")
    print_dict(parameters, report_file)

    results = text_parsing_results(a, text)
    print_dict(results, report_file)
    msg_to_logs("\nlayer #1 parsing results:\n\n", logs)
    map(log -> print_dict(results, log), logs)
    msg_to_logs("\n", logs)

    close(report_file)

    words_file = open(joinpath(report_path, "learned_words(all).txt"), "w")
    show_learned_words(a, words_file)
    close(words_file)

    words_file = open(joinpath(report_path, "learned_words(popular).txt"), "w")
    show_learned_popular_words(a, words_file)
    close(words_file)

    words_file = open(joinpath(report_path, "morphemes.txt"), "w")
    show_morphemes(a, 1, words_file)
    close(words_file)

    save_dir = mkpath(joinpath(app_path, "saved_objects", today, time))

    m_history = text_reparsing_history!(a, text, logs)

    msg_to_logs(string("morphemes after reparsing: ", count(x -> x > I(0), m_history), "\n"), logs)
    msg_to_logs(string("number of separators: ", count(x -> x == I(-2) || x == I(-3), m_history), "\n"), logs)
    msg_to_logs(string("number of unks: ", count(x -> x == I(-100), m_history), "\n"), logs)
    msg_to_logs(string("unique morphemes after reparsing: ", length(Set(m_history)), "\n"), logs)
    
    frequency_distribution(m_history, a.L, report_path)

    save_object(joinpath(save_dir, "L$(a.L)_m_history.jld"), m_history)
    save_object(joinpath(save_dir, "L$(a.L)_trained.jld"), a.layers[end])
    save_empty_adam!(a, save_dir)

    expand_options!(a)

    for ln ∈ a.L:(a.H - 1)

        trim_R!(a.layers[end], a.opt["min_counts"][a.L])
        init_layer!(a, m_history)
        save_object(joinpath(save_dir, "L$(a.L)_empty.jld"), a.layers[end])
        save_empty_adam!(a, save_dir)

        lp = a.layers[end-1]
        lp.stop_growth = true

        if a.L == 2
            base_words = Vector{String}(
                get(test_parameters, "base_words", ["February "]))
            find_synonyms(a, base_words, report_path)
        end
        find_full_synonyms(a, report_path, a.L)

        train_layer!(a, a.L, m_history, logs)
        m_history = reparse_morphemes!(a, a.L, m_history, logs)

        msg_to_logs(string("morphemes after reparsing: ", count(x -> x > I(0), m_history), "\n"), logs)
        msg_to_logs(string("number of separators: ", count(x -> x == I(-2) || x == I(-3), m_history), "\n"), logs)
        msg_to_logs(string("number of unks: ", count(x -> x == I(-100), m_history), "\n"), logs)
        msg_to_logs(string("unique morphemes after reparsing: ", length(Set(m_history)), "\n"), logs)
        
        frequency_distribution(m_history, a.L, report_path)

        save_object(joinpath(save_dir, "L$(a.L)_m_history.jld"), m_history)
        save_object(joinpath(save_dir, "L$(a.L)_trained.jld"), a.layers[end])

        if a.L == 2
            find_same_phrases(a, report_path)
        end
    end

    print_adam_statistics(a, report_path)
    end_time = Dates.now()
    msg_to_logs(string("program execution completed: ",
    Dates.format(end_time, "yyyy-mm-dd HH:MM:SS"),
    " ($(time_interval(start_time, end_time)))"), logs);

    return nothing
end
