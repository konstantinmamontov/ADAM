

encode(char::Char, a::Adam) = get(a.app["char -> l code"], char, M[])
encode(chars::Vector{Char}, a::Adam) = map(char -> encode(char, a), chars)
encode(str::String, a::Adam) = map(char -> encode(char, a), collect(str))

# decoding - by symbols of head №n

decode(s::M, a::Adam, n::Int=1)::String = 
    join([a.app["code -> char code"][n][k] 
    for k in a.layers[1].semantic[n].string[s]])
decode(w::Morpheme, a::Adam, n::Int=1)::String = 
    join(decode(s, a, n) for s in w)
decode(vw::Vector{Morpheme}, a::Adam, n::Int=1)::String =
    join(decode(w, a, n) for w in vw)

"""morpheme from layer #L plan decoding"""  
function decode(w::Morpheme, L::Int, a::Adam, n::Int=1)::String
    L <= 2 && return decode(w[n], a, n)

    l = a.layers[L-1]
    indices = rand(l.sequences[l.index[w]])

    return decode(map(i -> l.symbols[i], indices), L - 1, a, n)
end


"""morphemes from layer #L plan decoding"""  
decode(morphemes::Vector{Morpheme}, L::Int, a::Adam, n::Int=1)::String =
    join(map(w -> decode(w, L, a, n), morphemes))


function full_decode(w::Morpheme, L::Int, a::Adam, n::Int=1)::Vector{String}
    L <= 2 && return [decode(w[n], a, n)]

    l = a.layers[L-1]
    indices = collect(l.sequences[l.index[w]])

    return vcat(map(inds -> full_decode(map(i -> l.symbols[i], inds), L - 1, a, n), indices)...)
end

function full_decode(morphemes::Vector{Morpheme}, L::Int, a::Adam, n::Int=1)::Vector{String}
    if length(morphemes) == 1
        return full_decode(morphemes[1], L, a, n)
    end
    return vcat(map(str1 -> map(
        str2 -> join([str1, str2]), full_decode(morphemes[2:end], L, a, n)), full_decode(morphemes[1], L, a, n))...)
end