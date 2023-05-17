# Adaptive Deep Autonomous Machine (ADAM)  v.0.5.0                  # 2022-09-12
# Copyright (c) 2020-2023 Moscow Institute of Physics and Technology, Laboratory of Cognitive architectures


""" ~ DataStructures.counter replacement for v.0.5.0 planning consolidate(suggestions...) """
function counter(v::Vector{I})::Dict{I,Int}
    dict::Dict{I, Int} = Dict()
    for x in v
        dict[x] = get(dict, x, 0) + 1
    end
    return dict
end

""" ~ StatBase.countmap replacement to avoid StatBase pkg add requirement"""
function countmap(v::Vector)::Dict
    d = Dict(k => 0 for k in unique(v))
    for x in v
        d[x] += 1
    end
    return d
end

""" ~ Statistics.mean replacement to avoid Statistics pkg add requirement"""
function mean(x::Vector)
    sum(x) / length(x)
end

""" Check Layer.opt::Dict entry key => value : v == given if value::Bool else true if key exists """
function option(l::Layer, key::String, value::Bool=true)
    ∉(:opt, fieldnames(typeof(l))) ? false :
    !haskey(l.opt, key) ? false :
    typeof(l.opt[key]) == Bool ? l.opt[key] == value : true
end

""" Pretty print ::Dict """
function print_dict(dict::Dict, ostream=Base.stdout)
    if length(dict) > 0
        pairs = sort(collect(dict), by = x -> x[1])
        tab = maximum([length(x[1]) for x in pairs]) + 1
        clean_string(x) = replace(string(x), "Any"=>"")
        for p in pairs
            if p[2] isa Dict
                print(ostream, (string(p[1])*' '^tab)[1:tab]*"=> \n{\n")
                print_dict(p[2], ostream)
                print(ostream, "}\n")
            else
                print(ostream, (string(p[1])*' '^tab)[1:tab]*"=> "*clean_string(p[2])*"\n")
            end
        end
    end
end

"""
Print fields of a struct with values or short decriptions
# Agruments:
- `str::struct`: structure
- `short::Vector{symbol}`: struct fields to print description instead of value
- `indent::Int`: indentation - number of spaces before field value
- `nbr::Int`: number of linebreaks between fields
# Examples
    `print_fields(adam, [:layers])` - short description for the :layers field
"""
function print_fields(str, short=[], indent=0, nbr=1)::Nothing
    idt = indent > 0 ? ' '^indent : ""
    for (i,fn) in enumerate(fieldnames(typeof(str)))
        if fn in short
            print(idt, i,". ",fn, " => ", fieldtype(typeof(str),i))
        else
            print(idt, i,". ",fn, " => ", getproperty(str, fn))
        end
        for j=1:nbr print('\n') end
    end
    return nothing
end

""" Print a vector line by line """
function print_vec(vector)::Nothing
    for element in vector
        println(element)
    end
    return nothing
end

""" Save agent trial report from Tutorial.ipynb or the like tests """
function save_report(adam::Adam, parameters::Dict, data::Vector=[], title::String = "")
    p = get(parameters, "test_parameters", parameters)
    output_dir::String      = get(p, "output_dir", "")
    save_adam_backup::Bool  = get(p, "save_adam_backup", false)
    save_data::Bool         = get(p, "save_data", false) & !isempty(data)
    save_scatter_plot::Bool = get(p, "save_scatter_plot", true)
    save_sma_plots::Bool    = get(p, "save_sma_plots", true)
    save_sma_sum_plot::Bool = get(p, "save_sma_sum_plot", true)
    save_parameters::Bool   = get(p, "save_parameters", true)
    save_report::Bool       = get(p, "save_report", true)
    sw::Int                 = get(p, "sma_sampling_width", 10)
    verbose::Int            = get(p, "verbose", 1)

    (isempty(title)) && (title = Dates.format(Dates.now(), "yymmdd-HHMMSS"))
    report = Dict()
    if (save_parameters)
        path = joinpath(output_dir, title * "_parameters.json")
        report["parameters_path"] = path
        re_ = save_json(parameters, path)
        (verbose > 2) && print_dict(re_)
    end
    if (save_adam_backup)
        report["adam_backup_path"] = joinpath(output_dir, title * "_adam.jld")
        @time re_ = save_adam(adam, report["adam_backup_path"])
        (verbose > 2) && (println(); print_dict(re_))
    end
    if (save_data)
        d = Dict()
        (save_parameters)  && (d["parameters_path"] = report["parameters_path"])
        (save_adam_backup) && (d["adam_path"] = jld["path"])
        d["data"] = data
        d["path"] = joinpath(output_dir, title * "_data.jld")
        re_ = save_jld(d)
        report["data_path"] = re_["path"]
        (verbose > 2) && (println(); print_dict(re_))
    end
    report["plot_paths"] = []
    if (save_scatter_plot)
        plot_title = title * "_scatter.png"
        plot_path  = joinpath(output_dir, plot_title)
        p = plot(plot(data, seriestype = :scatter, title = plot_title, legend = false))
        savefig(plot_path)
        push!(report["plot_paths"], plot_path)
    end
    if (save_sma_plots)
        plot_title = title * "_sma_plots.png"
        plot_path  = joinpath(output_dir, plot_title)
        p = plot(map(x -> sma(x, sw), data), title = plot_title, legend = false)
        savefig(plot_path)
        push!(report["plot_paths"], plot_path)
    end
    if (save_sma_sum_plot)
        plot_title = title * "_sma_sum.png"
        plot_path  = joinpath(output_dir, plot_title)
        p = plot(mapreduce(x -> sma(x, sw), +, data)/length(data),
                 title = plot_title, legend = false)
        savefig(plot_path)
        push!(report["plot_paths"], plot_path)
    end
    if (save_report)
        report_path = joinpath(output_dir, title * "_report.json")
        re_ = save_json(report, report_path)
        (verbose > 2) && (println(); print_dict(re_))
    end
    return abspath(output_dir)
end

""" SMA: Simple Moving average """
function sma(series::Vector, sampling_width::Int)
    rolling_sum = 0
    sma_series = []
    rolling_mean = 0.
    for i = 1:length(series)
        rolling_sum += series[i]
        if i <= sampling_width
            rolling_mean = rolling_sum / i
        else
            rolling_sum -= series[i-sampling_width]
            rolling_mean = rolling_sum / sampling_width
        end
        push!(sma_series, Int(round(rolling_mean)))
    end
    return sma_series
end

""" Format time interval as a string: days, hours, minutes, seconds (no ms) """
function time_interval(start::Dates.DateTime, finish::Dates.DateTime)::String
    str = string(Dates.canonicalize(Dates.CompoundPeriod(finish - start)))
    vec = split(str, ",")
    return length(vec) < 2 ? str : join(split(str, ",")[1:end-1], ",")
end
