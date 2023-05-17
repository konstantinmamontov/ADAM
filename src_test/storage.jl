# Adaptive Deep Autonomous Machine (ADAM)  v.0.5.0                  # 2021-09-12
# Copyright (c) 2020-2023 Moscow Institute of Physics and Technology, Laboratory of Cognitive architectures

import JLD2, JSON

""" Save ::ADAM struct → .jld archive file, return report::Dict
    report["path"] - path to .jld file ("UTC_...jld" auto if no path parameter)
"""
function save_agent(a::Adam, path="")::Dict
    save_jld(Dict("agent" => a, "path" => path))
end

""" Save ::Dict → .jld archive file. Saves structs ~OK with JLD2 package """
function save_jld(d::Dict)::Dict
    datime = now(Dates.UTC)
    report = Dict("function" => "save_jld()", "datetime" => datime,
        "UTC" => Dates.format(datime, "yyyy-mm-dd_HH:MM:SS")*"_UTC")
    path = get(d, "path", "")

    if (length(path) < 1)
        filename = "UTC_"*Dates.format(datime, "yyyy-mm-dd_HH-MM-SS")*".jld"
        path = abspath(filename)
        if (isdir(dirname(abspath("")) * "/data"))
            path = dirname(abspath("")) * "/data" * filename
        end
    end
    d["path"] = path
    report["path"] = path

    try
        JLD2.jldopen(path, "w") do file
            for k in keys(d)
                JLD2.write(file, string(k), d[k])
            end
            JLD2.write(file, "datetime", report["datetime"])
            JLD2.write(file, "UTC", report["UTC"])
        end
        return report
    catch e
        @error "save_jld() error" => e
        report["error"] = e
        return report
    end
end

""" Load ::ADAM struct from a .jld archive file created using save_agent() function ↑ """
load_agent(path::String) = load_jld(path)["agent"]

""" Load .jld archive file → ::Dict """
function load_jld(path::String)::Dict
    @assert isfile(path)
    try
        data = JLD2.load(path)
        if (!haskey(data, "path"))
            push!(data, "path" => path)
        elseif data["path"] == ""
            data["path"] == path
        end
        return data
    catch e
        @error "load_jld() error" => e
        return Dict("function" => "load_jld(path)", "error" => e)
    end
end

""" Save ::Dict → .json file """
function save_json(json::Dict{}, path::String="", indent::Integer=4)::Dict
    datime = Dates.now()
    report = Dict(
        "function" => "save_json()",
        "datetime" => Dates.format(datime, "yyyy-mm-dd_HH:MM:SS")
    )
    if (length(path) < 1)
        filename = "save_" * Dates.format(datime, "yyyy-mm-dd_HH-MM-SS") * ".json"
        path = abspath(filename)
        if (isdir(dirname(abspath(""))*"/data"))
            path = dirname(abspath(""))*"/data" * filename
        end
    end
    report["path"] = abspath(path)

    try
        io = open(path, "w")
        if indent == 0
            JSON.print(io, json)
        else
            JSON.print(io, json, indent)
        end
        close(io)
        return report
    catch e
        @error "save_json() error" => e
        report["error"] = e
        return report
    end
end

""" Load .json file → ::Dict """
function load_json(path::String)::Dict
    report::Dict{String, Any} = Dict("function" => "load_json(path)", "path" => path)

    if !isfile(path)
        @error "load_json(path): wrong path" => path
        report["error"] = "wrong path"
        return report
    end

    try
        io = open(path, "r")
        json_string = read(io, String)
        close(io)
        return JSON.parse(strip_comments(json_string))
    catch e
        @error "load_json() error" => e
        report["error"] = e
        return report
    end
end

""" Delete comments from "augmented" JSON document """
function strip_comments(json::String)::String
    #-join([s for s in [strip_comment(s) for s in split(json, '\n')] if !isempty(s)], '\n')
    join(filter(s -> !isempty(s),
         map(s -> strip_comment(s), split(json, '\n'))),
         '\n')
end

""" Delete comments from "augmented" JSON document string """
function strip_comment(string::Union{String, SubString{String}})
    s = strip(string)
    if (isempty(s) || s[1] == '#')
        return ""
    end

    last_comma = findlast(",", s)
    if (isnothing(last_comma))
        if (occursin("#", s))
            return s[1 : findfirst("#", s)[1]-1]
        end
        return s
    end

    if (occursin("#", s[last_comma[1] : end]))
        return s[1 : last_comma[1]]
    end

    return s
end

