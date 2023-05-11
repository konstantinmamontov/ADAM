

function find_next_morphemes(a::Adam, report_path::String)::Nothing

    l = a.layers[2]

    report_file = open(joinpath(report_path, "L2_pairs.txt"), "w")

    for m ∈ eachindex(filter(w -> !isempty(w), l.morphemes))
        sequences = collect(l.sequences[m])
        next_sequences = vcat(map(m_ -> collect(l.sequences[m_]), l.memory[m])...)
        msg_to_logs("Морфема:\n", [report_file])
        for iv ∈ sequences
            wv = map(i -> l.symbols[i], iv)
            str = decode(wv, l.L, a)
            msg_to_logs("$(str)\n", [report_file])
        end
        msg_to_logs("Следующие морфемы:\n", [report_file])
        for iv ∈ next_sequences
            wv = map(i -> l.symbols[i], iv)
            str = decode(wv, l.L, a)
            msg_to_logs("$(str)\n", [report_file])
        end

    end

    close(report_file)

    return nothing
end

function print_adam_statistics(a::Adam, report_path::String)::Nothing

    for l ∈ a.layers
        print_layer_statistics(l, report_path)
        R_matrix_distribution(l, report_path)
    end

    return nothing
end

function print_layer_statistics(l::Layer, report_path::String)::Nothing

    report_file = open(joinpath(report_path, "L$(l.L)_layer_stats.txt"), "w")

    morphemes = filter(w -> !isempty(w), l.morphemes)
    #=probabilities = map(
        w -> mapreduce(h -> h.R_i[w[h.N]], *, l.semantic), morphemes)
    probabilities = probabilities ./ sum(probabilities)=#

    #=morphemes_sequences_avg = 0.0
    morphemes_length_avg = 0.0
    morphemes_next_avg = 0.0=#
    morphemes_sequences = Dict{Int, Int}()
    morphemes_length = Dict{Int, Int}()
    morphemes_next = Dict{Int, Int}()

    for m ∈ eachindex(morphemes)
        num = length(l.sequences[m])
        len = length(collect(l.sequences[m])[1])
        next = length(l.memory[m])
        #=morphemes_sequences_avg += num * probabilities[m]
        morphemes_length_avg += len * probabilities[m]
        morphemes_next_avg += next * probabilities[m]=#
        morphemes_sequences[num] = get(morphemes_sequences, num, 0) + 1
        morphemes_length[len] = get(morphemes_length, len, 0) + 1
        morphemes_next[next] = get(morphemes_next, next, 0) + 1
    end

    head_morphemes = map(h -> h.D, l.semantic)
    msg_to_logs("Число морфем голов: $(head_morphemes)\n", [report_file])
    msg_to_logs("Число морфем слоя: $(length(morphemes))\n", [report_file])
    sm_num = length(unique(l.symbol_morphemes))
    s_num = length(l.symbols)
    msg_to_logs("Число односимвольных морфем слоя: $sm_num\n", [report_file])
    msg_to_logs("Число символов слоя: $s_num\n", [report_file])
    msg_to_logs("Степень сжатия: $(s_num / sm_num)\n", [report_file])
    msg_to_logs("Число параметров: $(count_parameters(l))\n", [report_file])

    #msg_to_logs("Ожидаемое число реализаций морфем: $morphemes_sequences_avg\n", [report_file])
    msg_to_logs("Распределение количеств реализаций морфем:\n", [report_file])
    print_stats(morphemes_sequences, report_file)

    #msg_to_logs("Ожидаемая длина морфемы: $morphemes_length_avg\n", [report_file])
    msg_to_logs("Распределение длин морфем:\n", [report_file])
    print_stats(morphemes_length, report_file)

    #msg_to_logs("Ожидаемое количество продолжений морфемы: $morphemes_next_avg\n", [report_file])
    msg_to_logs("Распределение продолжений морфем:\n", [report_file])
    print_stats(morphemes_next, report_file)

    close(report_file)

    return nothing
end

function count_parameters(l::Layer)::Int

    parameters_num = 0
    parameters_num += mapreduce(h -> count_parameters(h), +, l.semantic)

    return parameters_num
end

function count_parameters(h::Head)::Int

    parameters_num = 0
    parameters_num += mapreduce(wh -> length(h.R[wh]), +, 1:h.D)

    return parameters_num
end

function R_matrix_distribution(a::Adam, report_path::String)::Nothing
    for l in a.layers
        R_matrix_distribution(l, report_path)
    end
    return nothing
end

function R_matrix_distribution(l::Layer, report_path::String)::Nothing
    for h in l.semantic
        R_matrix_distribution(h, report_path)
    end
    return nothing
end

function R_matrix_distribution(h::Head, report_path::String)::Nothing
    rv = R[]
    for row in h.R
        append!(rv, collect(values(row)))
    end
    
    histogram_path = joinpath(report_path, "L$(h.L)_h$(h.N)_R_histogram.pdf")

    histogram(log.(2, rv), yaxis=(:log10))
    title!("R counts")
    xlabel!("counts (log2)")
    ylabel!("elements")

    savefig(histogram_path)

    return nothing
end

function frequency_distribution(m_history::Vector{I}, L::Int, report_path::String)::Nothing

    counts = map(x -> F(x[2]), collect(countmap(m_history)))
    sort!(counts, rev = true)
    counts = cumsum(counts)

    counts ./= (counts[end] / 100.0)

    plot_file = joinpath(report_path, "L$(L)_history_coverage.pdf")

    plot(counts, legend = false)
    title!("History coverage by morphemes")
    xlabel!("Morphemes")
    ylabel!("History, %")

    savefig(plot_file)

    coverage_report = open(joinpath(report_path, "L$(L)_history_coverage_report.txt"), "w")

    for i in 10000:10000:length(counts)
        msg_to_logs(string("$(i) morphemes -> $(round(counts[i], digits = 2))%\n"), [coverage_report])
    end

    close(coverage_report)    

    return nothing
end


function frequent_morphemes(a::Adam, L::Int, m_history::Vector{I}, min_frequency::Int, report_path::String)::Nothing

    l = a.layers[L]

    report_file = open(joinpath(report_path, "L$(l.L)_frequent_morphemes.txt"), "w")
    m_c = collect(countmap(m_history))
    filter!(mc -> mc[2] > min_frequency, m_c)
    sort!(m_c, by = x -> x[2], rev = true)
    #m_c = sort(collect(countmap(m_history)), by = x -> x[2], rev = true)

    for (m, c) in m_c

        if m <= I(0)
            continue
        end

        strs = full_decode(l.morphemes[m], l.L + 1, a)

        msg_to_logs("$(m)\t$(c)\t$(length(strs))\t", [report_file])
        
        msg_to_logs("$(join(strs, "  "))\n", [report_file])
    end

    close(report_file)

    return nothing
end


        
