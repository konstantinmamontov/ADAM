# Adaptive Deep Autonomous Machine (ADAM)  v.0.5.3                  # 2022-12-12
# Copyright (c) 2020-2023 Moscow Institute of Physics and Technology, Laboratory of Cognitive architectures


""" Replace layer l head coders with new ones """
function implant_coder!(l::Layer, coders::Vector{Matrix{F}})::Vector{Matrix{F}}
    [implant_coder!(h, coders[h.N]) for h in l.semantic]
end

function implant_coder!(h::Head, coder::Matrix{F})::Matrix{F}
    h.coder = coder
    h.d = size(coder, 1)
    h.k = size(coder, 2)
    return h.coder
end

""" Create coders """
function new_coders(ex::Vector{Vector{F}}, d::Int, K::Vector{Int})::Vector{Matrix{F}}
    [new_coder(ex, d, k) for k in K]
end

""" Make a new coder for the 1st Layer in vector input environment """
function new_coder(ex::Vector{Vector{F}}, d::Int, k::Int)::Matrix{F}
    d == 2 ? kmeans_2d(ex, k) :
    kmeans(hcat(filter(x -> !isempty(x), ex)...), k).centers
end

function kmeans_2d(ex::Vector{Vector{F}}, k::Int)::Matrix{F}
    data = filter(x -> !isempty(x), ex)
    actions = unique(map(x -> x[1], data))
    d2 = length(actions)
    d1 = Int(floor(k / d2))
    tails = kmeans(hcat(map(x -> x[2:end], data)...), d1).centers
    coder = zeros(F, length(data[1]), d1*d2)
    i = 0
    for t = 1:size(tails)[2]
        for action in actions
            i += 1
            coder[1, i] = action
            coder[2:end, i] = tails[:, t]
        end
    end
    return coder
end

""" Make a new coder -- v0.5.1 legacy """
function new_coder(l::Layer)::Vector{Matrix{F}}
    map(k -> kmeans(hcat(clean_ex(l.episodic)...), k).centers, l.K)
end

""" Make a new coder for the 2+ layer v0.5.2 """
function new_coder(l::Layer, vectors::Matrix{F}, weights::Vector{F})
    #coder::Vector{KmeansResult} = []
    coder::Vector{Tuple{Matrix{F}, Vector{M}}} = []
    resize!(coder, l.N)
    #-println("56 D min_max = ", findmin(vectors), findmax(vectors))
    #-flush(Base.stdout)
    #vectors = hcat(vectors...)
    @threads for i in eachindex(l.K)
    #-for i in eachindex(l.K)
        if i == 1
            coder[i] = Kmeans_tree(vectors, l.K[i], weights)
        else
            coder[i] = Kmeans_tree(vectors, l.K[i], ones(F, length(weights)))
        end
        #coder[i] = Kmeans(vectors, M(l.K[i]), weights)
        #coder[i] = Kmeans(vectors, M(l.K[i]), ones(F, length(weights)))
    end
    #-map(k -> kmeans(hcat(vectors...), k, weights = weights), l.K)
    return coder
end

""" Prune episodic memory sequence of input vectors for a new coder creation """
function clean_ex(e::Episodic)::Vector{Vector{F}}
    [x for (i,x) in enumerate(e.x)
     if (e.t[i] > 0) && !isempty(x) && !any(isnan.(x))]
end

""" Find a layer l symbol and its index for a previous layer morpheme w sparse vector representaion x """
function encode!(l::Layer, s::Morpheme, x::Vector{Tuple{M, R}}=Tuple{M, R}[]
                 )::Tuple{I, I, Morpheme}
    if (isempty(s))
        return (I(0), I(0), M[])
    end
    if (haskey(l.symbol_index, s))
        i = l.symbol_index[s]
        m = l.symbol_morphemes[i]
        w = l.morphemes[m]
        return (i, m, w)
    end

    if (l.L == 1)
        w = s
    elseif (isempty(x))
        return (I(0), I(0), M[])
    else
        x_::Vector{F} = sparse2dense(x, l)
        if (isempty(x_))
            return (I(0), I(0), M[])
        end
        x_ = normalize(x_)
        w = encode(x_, l)
    end
    if (invalid(w))
        @warn "invalid w" => w
        return (I(0), I(0), M[])
    end

    i = I(length(l.symbols) + 1)
    m::I = add_layer_morpheme!(l, w, [i])
    if (m < I(1))
        return (I(0), I(0), M[])
    end

    push!(l.symbols, s)
    l.symbol_index[s] = i
    push!(l.symbol_morphemes, m)
    @assert I(length(l.symbols)) == I(length(l.symbol_morphemes)) == i

    return (i, m, w)
end

""" Convert a sparse morpheme representation vector to a dense embedding """
function sparse2dense(x::Vector{Tuple{M,R}}, l::Layer)::Vector{F}
    result = zeros(F, l.d)
    for (i, r) in x
        for j in 1:l.d
            if l.Rnd[j, i]
                dv[j] += r
            else
                dv[j] -= r
            end
        end
    end
    return result
end

function sparse2dense!(dv::Vector{F}, x::Vector{Tuple{M,R}}, l::Layer)::Nothing
    for (i, r) in x
        for j in 1:l.d
            if l.Rnd[j, i]
                dv[j] += r
            else
                dv[j] -= r
            end
        end
    end
    return nothing
end

function dense_vector(subtotals::Vector{Vector{Vector{F}}}, w::Morpheme)::Vector{F}
    dv = zeros(F, length(subtotals[1][1]))

    for hN ∈ eachindex(subtotals)
        axpy!(1.0, subtotals[hN][w[hN]], dv)
    end
    len = norm(dv)
    if len < 1.0e-6
        dv = rand(F, length(dv))
        normalize!(dv)
    else
        dv ./= len
    end

    return dv
end

function dense_vector!(dv, subtotals::Vector{Vector{Vector{F}}}, w::Morpheme)::Nothing

    for hN ∈ eachindex(subtotals)
        axpy!(1.0, subtotals[hN][w[hN]], dv)
    end
    len = norm(dv)
    if len < 1.0e-6
        dv = rand(F, length(dv))
        normalize!(dv)
    else
        dv ./= len
    end

    return nothing
end

""" Normalize a real number vector x → morpheme """
#=function normalize(x)::Vector{F}
    n = norm(x)
    if n == 0.0
        return x
    end
    return x ./ n
end=#

""" Encode a real number vector x → morpheme : list of layer head cluster numbers """
function encode(x::Vector{F}, l::Layer)::Vector{M}
    if (isempty(x))
        return M[]
    end
    return map(h -> encode(x, h), l.semantic)
end

""" Find a head h coder cluster most similar to a real vector number x """
function encode(x::Vector{F}, h::Head)::M
    M(findmin(map(k -> norm(x - h.coder[:,k]), 1:h.k))[2])
end

""" Create a new layer morpheme index instance for a morpheme w """
function add_layer_morpheme!(l::Layer, w::Vector{M}, ī::Vector{I})::I
    if (haskey(l.index, w))
        m = l.index[w]
        if (ī ∉ l.sequences[m])
            push!(l.sequences[m], ī)
        end
        return m
    end
    m::I = l.n + I(1)
    if (m > l.W)
        return I(0)
    end
    l.n = m
    l.index[w]     = m
    l.morphemes[m] = w
    l.sequences[m] = Set([ī])
    for n ∈ 1:l.N
        push!(l.semantic[n].m̄[w[n]], m)
    end
    return m
end

""" Create a sparse morpheme representation vector """
function sparse_vector(w::Morpheme, l::Layer)::Vector{Tuple{M,R}}
    if (isempty(w))
        return Tuple{M,R}[]
    end
    return vcat(map(h -> head_vector(w[h.N], h, M(2*l.M*(h.N-1))), l.semantic)...)
end

function head_vector(m::M, h::Head, offset::M=M(0))::Vector{Tuple{M,R}}
    filter(x -> x[2] > h.ppmi_threshold, [
        map(j -> (j+offset, pmi(m,j,h)), collect(keys(h.R[m])));
        map(i -> (i+offset+M(h.M), pmi(i,m,h)), collect(keys(h.Rt[m])))])
end

""" Positive pointwise mutual information """
function pmi(i::M, j::M, h::Head)::R
    return log((h.R[i][j] + 1) * (h.R_if + 1) / ((h.R_i[i] + 1) * (h.R_f[j] + 1)))
    #pmi = log((h.R[i][j] + 1) * (h.R_if + 1) / ((h.R_i[i] + 1) * (h.R_f[j] + 1)))
    #if (pmi > h.ppmi_threshold)
    #    return pmi
    #end
    #return R(0.)
end

""" Decode layer morphemes based plans → symbol based (previous layer morphemes) """
function decode(plans::Vector{Plan}, l::Layer)::Vector{Plan}
    vcat(map(plan -> decode(plan, l), plans)...)
end

""" Decode a single w morpheme based plan to a list of symbol-based plans """
function decode(plan::Plan, l::Layer)::Vector{Plan}
    (w̄, v,p,t) = plan
    if (isempty(w̄) || isempty(w̄[1]))
        return Plan[]
    end
    if (length(w̄) > 1)
        return decode_long_plan(plan, l)
    end
    w::Vector{M} = w̄[1]
    s″::Vector{Vector{Morpheme}} = Vector{Morpheme}[]
    # Decode fast
    if (l.planner.decode_fast)
        s″ = decode_fast(w, l)
    end
    # Decode slow
    if ((length(s″) < l.planner.min) && l.planner.decode_slow)
        append!(s″, decode_slow(w, l))
    end
    # Decode only the 1st 1-symbol morpheme (backup)
    if (length(s″) < l.planner.min)
        w″ = decompose(w̄[1], l)
        if (!isempty(w″))
            append!(s″, decode_fast(w″[1], l))
        end
    end
    return map(w̄′ -> (w̄′, v,p,t), s″)
end

""" Decode plan with more than 1 morpheme:stub, not used """
function decode_long_plan(plan::Plan, l::Layer)::Vector{Plan}
    stub = Tuple([[plan[1][1]]; plan[2:end]])
    return decode(stub, l)
end

function decode_fast(w::Vector{M}, l::Layer)::Vector{Vector{Morpheme}}
    if (invalid(w) || !haskey(l.index, w))
        return Vector{Morpheme}[]
    end
    i″::Vector{Vector{I}} = collect(l.sequences[l.index[w]])
    s″::Vector{Vector{Morpheme}} = map(ī -> map(i -> l.symbols[i], ī), i″)
    return s″
end

function decode_slow(w::Vector{M}, l::Layer)::Vector{Vector{Morpheme}}
    m̄::Vector{I} = map(x -> get(l.index, x, I(0)), decompose(w, l))
    if (any(x -> x < I(1), m̄))
        return Vector{Morpheme}[]
    end
    i″::Vector{Vector{I}} = map(m -> map(ī -> ī[1], collect(l.sequences[m])), m̄)
    return mix(map(i -> l.symbols[i], i″))
end

function mix(w″::Vector{Vector{Morpheme}})::Vector{Vector{Morpheme}}
    (isempty(w″) || any(x -> isempty(x), w″)) ? Vector{Morpheme}[] :
    length(w″) == 1 ? map(x -> [x], w″[1]) :
    [[[x]; y] for x ∈ w″[1] for y ∈ mix(w″[2:end])]
end

""" Decompose morpheme sequences in plans to simple morheme sequences """
function decompose(plans::Vector{Plan}, l::Layer)::Vector{Plan}
    map(plan -> decompose(plan, l), plans)
end

function decompose(p::Plan, l::Layer)::Plan
    isempty(p[1]) ? (Morpheme[], R(0.),  R(0), T(0)) :
    (decompose(p[1], l), p[2], p[3], p[4])
end

function decompose(w̄::Vector{Morpheme}, l::Layer)::Vector{Morpheme}
    length(w̄) == 1 ? decompose(w̄[1], l) :
    vcat(map(w -> decompose(w, l), w̄)...)
end

function decompose(w::Morpheme, l::Layer)::Vector{Morpheme}
    if (isempty(w) || any(map(x -> x < 1, w)))
        return Morpheme[]
    end
    return mapslices(x -> [x],
           reduce(hcat, map(h -> h.string[w[h.N]], l.semantic)), dims=2)[:]
end

#--------------------------------------
function Kmeans(D, m_num::M, weights, iter_num::Int = 50,
  dI_min::F = F(1e-3), NORM::Int = 2, REPORT=true)#::Tuple{Vector{M}, Vector{F}}
    # Boost K-means ---------------------
    # Input:
    # D[c,f] (Datapoints = normalized columns)
    # m_num - number of clusters
    # Output:
    # cluster[f] - clusters for features
    function choose_cluster(m0::M, Prob::Matrix{F}, Clusters::Vector{Bool})::M
      # Prob[i,j] -> Estimated Probability to change i -> j
      P_max = 0
      m = 1
      for k in eachindex(Clusters)
        if Clusters[k]
          P = Prob[m0,k]
          if P > P_max
            P_max = P
            m = k
          end
        end
      end
      return m
    end
    function compare_clusters_0(f::Int, m0::M, m::M, D, Dm::Matrix{F}, W::Vector{F}, Im::Vector{F}, wf::F)::Tuple{F, F, F}
      # Value for 'f' to switch from m0 -> m
      # For non-normalised data in L2
      # I = |D'D|/N (dist between members of a cluster)
      @views Im_new = (Im[m] * W[m] + 2.0 * wf * dot(D[:,f], Dm[:,m]) + wf ^ 2) / (W[m] + wf)
      @views Im0_new = (Im[m0] * W[m0] - 2.0 * wf * dot(D[:,f], Dm[:,m0]) + wf ^ 2) / (W[m0] - wf)
      delta_I = Im_new + Im0_new - Im[m] - Im[m0]
      return delta_I, Im0_new, Im_new
    end
    function compare_clusters_1(f::Int, m0::M, m::M, D, Dm::Matrix{F}, LDm::Matrix{F}, Nm::Vector{Int}, Im::Vector{F})::Tuple{F, F, F}
      # Value for 'f' to switch from m0 -> m
      # For L1-NORMALIZED data ONLY !!!
      # I = D'log(C) (KL-dist between members D and centroid C of a cluster)
      #=
      delta_Im = -sum(D[:,f].*LDm[:,m])/Nm[m]
      delta_Im0 = sum(D[:,f].*LDm[:,m0])/(Nm[m0]-1)
      delta_I = delta_Im + delta_Im0
      return delta_I, Im[m0]+delta_Im0, Im[m]+delta_Im
      #
      Ld = log(D[:,f]+eps(Float32))
      delta_Im = -sum(Dm[:,m].*Ld + D[:,f].*LDm[:,m])/(Nm[m]+1)
      delta_Im0 = sum(Dm[:,m0].*Ld+D[:,f].*LDm[:,m0])/(Nm[m0]-1)
      delta_I = delta_Im + delta_Im0
      return delta_I, Im[m0]+delta_Im0, Im[m]+delta_Im
      =#
      #
      Ld = log(D[:,f]+eps(F))
      Im_new = Im[m]*(Nm[m]+1) - sum(Dm[:,m].*Ld + D[:,f].*(LDm[:,m]+Ld))
      Im_new /= (Nm[m]+1)
      Im0_new = Im[m0]*(Nm[m0]-1) + sum(Dm[:,m0].*Ld + D[:,f].*(LDm[:,m0]-Ld))
      Im0_new /= (Nm[m0]-1)
      delta_I = Im_new + Im0_new - Im[m] - Im[m0]
      return delta_I, Im0_new, Im_new
      #
    end
    function compare_clusters_2(f::Int, m0::M, m::M, D, Dm::Matrix{F}, Nm::Vector{Int}, Im::Vector{F}, wf::F)::Tuple{F, F, F}
      # Value for 'f' to switch from m0 -> m
      # For L2-NORMALIZED data ONLY !!!
      # I = sqrt(|D'D|) (dist between members and centroid of a cluster)
        #-Im_new = sqrt(Im[m]^2 + 2*sum(D[:,f].*Dm[:,m]) + 1)
        #-Im0_new = sqrt(Im[m0]^2 - 2*sum(D[:,f].*Dm[:,m0]) + 1)
        #-Im_new = sqrt(Im[m]^2 + 2 * mapreduce(i -> D[i,f] * Dm[i,m], +, 1:(size(D)[1])) + 1)
        #-Im0_new = sqrt(Im[m0]^2 - 2 * mapreduce(i -> D[i,f] * Dm[i,m0], +, 1:(size(D)[1])) + 1)
        @views Im_new = sqrt(Im[m]^2 + 2.0 * wf * dot(D[:,f], Dm[:,m]) + wf ^ 2)
        @views Im0_new = sqrt(Im[m0]^2 - 2.0 * wf * dot(D[:,f], Dm[:,m0]) + wf ^ 2)
        delta_I = Im_new + Im0_new - Im[m] - Im[m0]
      return delta_I, Im0_new, Im_new
    end
    #-------------------------------------------------------------------------------
    #D = float(D)
    (c_num, f_num) = size(D)
    if REPORT
      println("K-means: Arranging ", f_num, " $c_num-dim data points in ", m_num, " clusters")
    end
    #-println("402 D min_max = ", findmin(D), findmax(D))
    #-flush(Base.stdout)
  
    # Initialize clustering:
    cluster::Vector{M} = rand(M(1):m_num, f_num)
    #for f=1:f_num
    #  cluster[f] = rand(1:m_num)  # cluster[f] = random m
    #end
  
    # Initialize Probabilities
    Prob = zeros(F, m_num,m_num)
  
    # Compute Dm = sum(f in m) for each cluster & I - functional to be maximized
    I = F(0.0)
    Dm = zeros(F, c_num,m_num)
    LDm = zeros(F, c_num,m_num)
    Nm = zeros(Int, m_num)
    W = zeros(F, m_num)
    Im = zeros(F, m_num)
    for m=1:m_num
      features = findall(cluster .== m)  # 131
      Nm[m] = length(features)
      if Nm[m] > 0
        #-@views sum!(Dm[:, m], D[:, features])
        for f in features
          @views axpy!(weights[f], D[:, f], Dm[:, m])
          W[m] += weights[f]
        end
        #-println("Dm = ", Dm[:, m])
        #-flush(Base.stdout)
        #-Dm[:, m] = mapreduce(f -> D[:,f], +, features)
        #-for f in features
        #-  Dm[:,m] += D[:,f]
        #-end
        if NORM == 2 # Cos-distance in L2
          # I = sqrt(|D'D|) (dist between members and centroid of a cluster)
          Im[m] = sqrt(mapreduce(i -> Dm[i,m] ^ 2, +, 1:c_num))
        elseif NORM == 1 # KL-divergence in L1
          # I = mean KL-dist between all members of a cluster (too time consuming!!!)
          for f in features
            LDm[:,m] += log(D[:,f]+eps(Float32))
          end
          Im[m] = -sum(Dm[:,m].*LDm[:,m])/Nm[m] # Proxy !!!
        else # Unnormalised L2 data
          # I = |D'D|/N (dist between members of a cluster)
          Im[m] = mapreduce(i -> Dm[i,m] ^ 2, +, 1:c_num) / W[m]
        end
        #
        I += Im[m]
      end
    end
    #-println("I = ", I)
    #-flush(Base.stdout)
    # MAIN LOOP ------------------------------------------------------------------
    time0 = time()
    Clusters::Vector{Bool} = falses(m_num)
    ff::Vector{Int} = collect(1:f_num)
    delta_I::F = 0.0
    Im0_new::F = 0.0
    Im_new::F = 0.0
    for iter = 1:iter_num
      t0 = time()
      dI = 0.0
      # Initialize Prob for current iteration
      for i=1:m_num, j=1:m_num
        if i != j Prob[i,j] = 1.0/(m_num+rand()) end
      end
      # Assign clusters for each feature -----------------------------------------
      ff = shuffle!(ff)
      for f in ff
        m0 = cluster[f]
        if Nm[m0] <= 1
          # Skip empty or singleton clusters
          continue
        end
         # All clusters
         # Except initial cluster
        fill!(Clusters, true)
        Clusters[m0] = false
        while any(Clusters)
          m = choose_cluster(m0,Prob,Clusters)
          if NORM == 2
            delta_I, Im0_new, Im_new = compare_clusters_2(f,m0,m,D,Dm,Nm,Im, weights[f])
          elseif NORM == 1
            delta_I, Im0_new, Im_new = compare_clusters_1(f,m0,m,D,Dm,LDm,Nm,Im)
          else
            delta_I, Im0_new, Im_new = compare_clusters_0(f,m0,m,D,Dm,W,Im, weights[f])
          end
  
          #println("Try: ", m0," -> ",m, " delta_I = ", round(delta_I, digits=2)) # DEBUG!!!
          if delta_I > 0.0
            # Change cluster for feature 'f'
            cluster[f] = m
            # Extract feature from cluster m0
            #-@views Dm[:,m0] .-= D[:,f]
            @views axpy!(-weights[f], D[:,f], Dm[:,m0])
            if NORM == 1 LDm[:,m0] -= log(D[:,f]+eps(Float32)) end
            Nm[m0] -= 1
            W[m0] -= weights[f]
            Im[m0] = Im0_new
            # Add feature to cluster m
            #-@views Dm[:,m] .+= D[:,f]
            @views axpy!(weights[f], D[:,f], Dm[:,m])
            if NORM == 1 LDm[:,m] += log(D[:,f]+eps(Float32)) end
            Nm[m] += 1
            W[m] += weights[f]
            Im[m] = Im_new
            dI += delta_I
            #println("     Success for $f -> dI = ",round(dI, digits=2))
            Prob[m0,m] += 1.0   # Update Probability
            break
          else
            # Try another cluster
            Clusters[m] = false
          end # m0 -> m trial
        end # clusters-loop
      end # f-loop
      I += dI
      if REPORT
        dt = time()-t0
        println("K-means-iter ",iter, ": dI/I = ",round(dI/I, digits=5)," -> ", round(dt, digits=2), " sec")
      end
      if dI/I < dI_min
      if REPORT println("dI/I = ", round(dI/I, digits=5), " < ",dI_min) end
        break
      end
      flush(Base.stdout)
    end # iter-loop
    # Report: -------------------------------------------
    if REPORT
      dtime = time()-time0
      println("---------------------------------")
      println(" K-means for ",m_num," clusters in ",round(dtime, digits=2)," sec ")
    end
    # ---------------------------------------------------
    return cluster, Im
    #Dm ./= W'
    #return (Dm, cluster)
  end
  
  #--------------------------------------
  function Kmeans_tree(D, m_num::Int, weights::Vector{F} = F[],
    iter_num::Int = 50, dI_min::F = F(1e-4), NORM::Int = 2, SPLIT_LARGEST::Bool = true)::Tuple{Matrix{F}, Vector{M}}
    # Boost K-means with clusters bisection ------------------
    # Input:
    # D[c,f] (Datapoints = columns) - Transposed D[f,c] from prepare_PLSA_data
    # m_num - number of clusters
    # SPLIT_LARGEST = true - Split largest cluster (else - cluster with largest mean error)
    # Output:
    # cluster[f] - clusters for features
    #-D = float(D)
    #-println("553 D min_max = ", findmin(D), findmax(D))
    #-flush(Base.stdout)
    (c_num, f_num) = size(D)
    if length(weights) != f_num
        weights = ones(F, f_num)
    end
    println("K-means tree: Arranging ", f_num, " $c_num-dim data points in ", m_num, " clusters")
  
    # Initial clustering:
    cluster = ones(M,f_num)
    cluster_m = zeros(M,f_num)
    I_m::Vector{F} = [0.0, 0.0]
  
    # MAIN LOOP ------------------------------------------------------------------
    time0 = time()
    N = zeros(M,m_num)
    W = zeros(F, m_num)
    Im = zeros(F, m_num)
    N[1] = length(cluster)
    W[1] = sum(weights)
    k_max = 0
    for m = 1:m_num-1
      t0 = time()
      #-k_max = m
      if m == 1 || SPLIT_LARGEST
        # Choose the largest cluster: All clusters ~ of the same size
        #(N_max,k_max) = findmax(N[1:m])
        k_max = 0
        w_max = F(0.0)
        for k in 1:m
          if N[k] > 1 && W[k] > w_max
            k_max = k
            w_max = W[k]
          end
        end
      else  # SPLIT_LARGEST_ERROR
        # Choose cluster with largest mean error (E/N): Large dense clusters
        # (E_max,k_max) = findmax(N[1:m].^2 - Im[1:m].^2) # Largest sum error
          (E_max,k_max) = findmin( Im[1:m]./(N[1:m] + eps(Float16)) )  # Largest mean error
      end
  
      # Bisect largest cluster with K-means
      ff = findall(cluster .== k_max) # 131
      REPORT=false
      #@views cluster_m, I_m = Kmeans(D[:,ff], M(2), weights[ff], iter_num, dI_min, NORM, REPORT)
      @views cluster_m, I_m = Kmeans(D[:,ff], M(2), ones(F, length(ff)), iter_num, dI_min, NORM, REPORT)
      #cluster_m, I_m = kmeans(D[:,ff], 2, weights = weights[ff], maxiter = 30, tol = 1e-4).assignments, [0.0, 0.0]
      for f in eachindex(ff)
        if cluster_m[f] != 1
          cluster[ff[f]] = m+1
          N[k_max] -= 1
          W[k_max] -= weights[ff[f]]
          N[m+1] += 1
          W[m+1] += weights[ff[f]]
        end
      end
      Im[k_max] = I_m[1]
      Im[m+1] = I_m[2]
  
      # Report:
      dt = time()-t0
      #println("---------------------------------")
      #println("K-means ",m+1, "-tree: ($(N[k_max])+$(N[m+1])) -> ", round(dt, digits=2), " sec")
      println("K-means ",m+1, "-tree: ($(N[k_max])+$(N[m+1])) -> ", round(dt, digits=2), " sec, Im = ", I_m)
      flush(Base.stdout)
      #
    end # m-loop
    # Clustering quality: max(I = sqrt(|D'D|))
    I = 0.0
    E = 0.0
    coder = zeros(F, c_num, m_num)
    for m=1:m_num
      Dm = zeros(F, c_num, 1)
      features = findall(cluster .== m)
      @views sum!(Dm[:, 1], D[:, features])
      if !isempty(features)
        @views coder[:, m] = Dm[:, 1] / length(features)
      end
      # I = sqrt(|D'D|) (dist between members and centroid of a cluster) -------
      I += sqrt(sum(Dm.*Dm))
      E += N[m]^2 - Im[m]^2
    end
    # Report: -------------------------------------------
    dtime = time()-time0
    println("===================================================================")
    println("I= ",I," f_num= ",f_num," E= ",E)
    println("K-means tree for $m_num clusters: I/N = ", round(I/f_num, digits=2)," E/N =", round(sqrt(abs(E))/f_num, digits=2)," -> ",round(dtime, digits=2)," sec")
    # RETURN:
    return coder, cluster
  end