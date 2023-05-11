# Adaptive Deep Autonomous Machine (ADAM)  v.0.5.3                  # 2022-12-12
# Copyright (c) 2020-2022 Oleg Baskov https://github.com/OlegBaskov

const C = Float64   # R-matrix count type
const Float = Float32
const F = Float32   # Coder, Episodic.x ::Vector{Vector{F}
const I = Int32       # Index type ~ size_t
const M = Int32       # Symbol type » Morpheme
const P = Float32   # Probability type  #TODO → Float32
const R = P         # Reward type = probability type
const T = Int       # TimeTick
const V = R         # Value type = reward
const Morpheme = Vector{M}
const Path  = Tuple{Vector{I}, R, R}
const Plan  = Tuple{Vector{Morpheme}, R, R, T}
const Prediction = Tuple{Vector{I}, P, V, T}


mutable struct Episodic     # 0.5.0 2022-07-25
    length::Int             # Episodic memory record lendth
    e::Vector{Tuple}        # Sensoric input, e.g. (a,s,r) - action, state, reward
    i::Vector{I}            # Layer symbol index of the input morpheme w
    m::Vector{I}            # Layer morpheme index
    r::Vector{R}            # External reward / lower layer reward
    s::Vector{Morpheme}     # Layer alphabet symbol vector ← encoded .x
    t::Vector{T}            # Time labels - sync with the lower layer and/or environment
    w::Vector{Morpheme}     # Input: the lower layer morphemes (for layers L>1)
    x::Vector{Vector{F}}    # Vector representation of the input morpheme w
end
Episodic() = Episodic(0, [], [], [], [], [], [], [], [])

mutable struct Head         # 0.5.0 2022-09-05
    L::Int                  # Layer number
    N::Int                  # Head number
    k::Int                  # Number of clusters
    d::Int                  # Dimensionality: L1: length(actions) + length(state)
    D::Int                  # Number of morphemes
    M::Int                  # Max number of morphemes
    adaptive_clusters::Bool # update coder cluster centroids upon each request
    cluster_mass::R         # initial cluster mass
    coder::Matrix{F}        # Coder cluster centroids
    discount::R             # morpheme value discount : previous ⇐ next
    l::Int                  # Max morpheme length
    learning_rate::R        # value update increment rate

    m̄::Vector{Set{I}}       # layer morpheme indices relating to the head morpheme
    max_plans::Int          # max number of plans to choose from R matrix
    n::Vector{Int}          # number of morpheme mean reward (Head.r) updates
    ppmi_threshold::R       # "positive" PMI threshold (default zero)
    r::Vector{R}            # morpheme mean reward
    R::Vector{Dict{M, R}}   # morpheme co-occurrence count matrix
    Rt::Vector{Dict{M, R}}  # R transposed
    R_i::Vector{C}          # R row sum
    R_f::Vector{C}          # R column sum
    R_if::C                 # R matrix sum

    sampling_dispersion::R  # Gaussian sampling dispersion
    semantic_threshold::R   # min reward for semantic memory growth
    string::Vector{Vector{M}}       # morpheme decomposition w ⇒ [s] vector of symbols
    string_idx::Dict{Vector{M}, M}  # string index [s] ⇒ w
    thompson_sampling::String       # Sampling algorithm : "Poisson" / "Gauss"
    v::Vector{Vector{R}}    # values: [[max, avg, mean, custom, count]]
    values::Int             # value-based action choice criteria: 1:max, 2:average, 3:mean
    value_threshold::R      # minimal value to update predecessor values
    w::Dict{Tuple{M, M}, M} # morpheme pair merges (w1,w1) ⇒ w′
end
Head() = Head(  0, 0, 0, 0, 0, 0, false, R(0.), zeros(F,1,1), R(0.), 0, R(0.),
                [], 100, [], R(0.), [], [], [], [], [], R(0),
                R(0.33), R(0.0), [], Dict(), "", [], 2, R(0.), Dict()
              )

mutable struct Parser       # 0.5.1 2022-09-15
    O::Int                  # parser order
    out::Tuple{Morpheme, R, T}  # parser output: (w_, r_, t_)
    last_merge::Int         # Last merged morpheme index in the stack, 0 : no merge in the last tick
    last_merged_morpheme::Tuple{Morpheme, R, T}
    stack::Vector{Tuple{Morpheme, R, T, Vector{I}}}  # [(w, r, t, ī)]
    r::Vector{R}            # reward stack
    memorise::Bool          # use memory (internal use, set parameter memory_length = 0 to avoid)
    memory:: Matrix{M}      # memorised morphemes
    memory_full::Bool       # value update process flag
    memory_length::Int      # = adam.working_memory[L]
    recent::Int             # last  morpheme in the memory
    reward::Vector{R}       # memorised morpheme rewards
end
Parser() = Parser(  2, (M[], R(0.), T(0)), 0, (M[], R(0.), T(0)),
                    Tuple{Morpheme, R, T, Vector{I}}[],
                    R[], true, zeros(M, 0, 0), false, 0, 0, R[]
                  )

mutable struct Planner      # 0.5.1 2022-10-21
    decode_fast::Bool       # decode a (long) layer morpheme using Layer.sequences memory
    decode_num::Int         # max number of plans to be returned to the lower layer
    decode_slow::Bool       # decode morphemes as a cartesian product of symbols
    instant_plans::Vector{Plan}  # instant (layer-level) plans stash
    last_morpheme::Morpheme
    max::Vector{Int}        # max number of plans chosen from any head R matrix
    min::Int                # min number of plans valid for head voting results
    next_symbol::Morpheme
    order::Int              # 1+: use all intermediate stack morphemes for predictions
    planning_fast::Bool
    planning_slow::Bool
    plans::Vector{Plan}     # upper layer plans (directives) stash
    quorum::Float           # head voting quorum (= number of heads if more)
    validate_plans::Bool    # check plan compliance with episodic memory
end
Planner() = Planner(true, 1, true, Plan[], M[], Int[], 1, M[],
                    2, false, true, Plan[], 0.51, true
                    )

mutable struct Layer        # 0.5.1 2022-09-30
    L::Int                  # Layer number
    M::Int                  # Max number of morphemes in layer heads
    N::Int                  # Number of heads
    K::Vector{Int}          # Number of clusters in the layer heads
    W::Int                  # Max number of layer morphemes
    adaptive_clusters::Bool # update coder cluster centroids upon each request
    alphabet::Vector{Char}  # FIXME?
    clustering::String      # Clustering algorithm: "kmeans", "boost_kmeans"
    cluster_mass::Vector{R} # Initial mass of a cluster upon creation
    coder::Int              # L1: 1  ⇒ 1D Kmeans; 2 ⇒ [a,o] experimental; L>1  ⇒ reduced space dim
    d::Int                  # Dimensionality: L1: actions+sensors; L>1: clustering space dim
    discount::Vector{R}     # Value learning discounts for the layer heads
    encoder::Dict{Char, M}  # Symbolic layer input encoder
    episodic::Episodic      # Episodic memory
    episodic_threshold::Int # Size of episodic memory to start a new upper layer
    fitness::Int            # Plans sorting criterion: 2 - by value, 3 - by probability assesment
    global_value_update::Bool  # Choose global vs local morpheme value update
    index::Dict{Morpheme,I} # Layer morpheme index
    initial_value::R        # Initial value of a morpheme upon creation
    input_m::Dict{Vector{M}, I}     # w → m::I input morpheme → layer morpheme index
    inputs::Vector{Set{Vector{M}}}  # s::I → ([w]) layer symbol → input morphemes
    internal_reward::R      # Internal reward for the predicted symbol
    l::Int                  # Max morpheme length
    learning_rate::Vector{R}# α - value update rate
    memory::Vector{Set{I}}  # Layer morpheme successors
    min_plan_length::Int    # Minimal plan length, Planning 0.5 lₚₗₐₙ
    morpheme_symbols::Vector{Vector{I}}  # Layer morpheme representations as strings of layer symbol indices
    morphemes::Vector{Morpheme}  # Layer morphemes
    n::I                    # Layer morphemes number
    next_layer_symbol::Vector{Bool} # Flag: layer morpheme is the next layer symbol
    parser::Parser
    planner::Planner
    ppmi_threshold::Vector{R} # Positive PMI threshold (default zero, reserved)
    Rnd::BitMatrix          # Random number matrix for dimensionality reduction

    sampled_values::Dict{Vector{Morpheme}, R}  # Plan values sampled once during epoch
    sampling_dispersion::R  # Gaussian sampling dispersion
    semantic::Vector{Head}  # Semantic memory : array of heads
    semantic_threshold::Vector{R}   # min reward for semantic memory growth
    sequences::Vector{Set{Vector{I}}}  # Layer morpheme projections to the lower layer
    stop_growth::Bool       # Flag to stop new morpheme creation after the next layer creation
    step_reward::R          # reward > 0 / penalty < 0 for each step (move)
    symbols::Vector{Morpheme}       # Layer symbols
    symbol_index::Dict{Morpheme,I}  # Layer symbol index
    symbol_morphemes::Vector{I}     # Layer symbol → Layer morpheme index
    sync_output::Bool       # Update response plans to fit current time tick (cut the past)
    thompson_sampling::String # Sampling algorithm : "Poisson", "Gauss"
    threading::Bool         # Use multi-threading calling the upper layer

    update_rewards::Bool
    update_values::Bool
    value_threshold::R      # min value to account for
    values::Vector{Int}     # value based plan choice criteria (head-wise):
                            # 1 : max, 2 : incrementally updated (learned) average
    winner_pair_count::R    # Head.R ⇑ for the winning pair of morphems in the parser stack
    opt::Dict               # development options
end
Layer() = Layer(0, 0, 0, Int[], 0, false, Char[], "", [R(0.)], 1, 0, R[],
                Dict(), Episodic(), 0, 2, false, Dict(), R(0.), Dict(), [],
                R(1.), 0, R[], Set{I}[], 1, Vector{I}[], Vector{M}[], I(0),
                Bool[], Parser(), Planner(), [R(0.)], falses(1,1),
                Dict(), 0.33, Head[], R[], Set{Vector{I}}[], false, R(0.),
                Morpheme[], Dict(), I[], false, "", true,
                true, true, R(0.1), R[], R(1), Dict()
                )

mutable struct Adam         # 0.5.1 2022-09-30
    H::Int                  # Maximum number of layers
    L::Int                  # Current number of layers
    M::Vector{Int}          # Max number of morphemes in layer heads
    N::Vector{Int}          # Number of heads in layers
    K::Vector{Vector{Int}}  # Number of clusters in layer modules
    O::Vector{Int}          # Parser order(s) for layers
    W::Vector{Int}          # Max number of layer morphemes
    application::String     # "MountainCarV0", "CartPoleV1", ...
    adaptive_clusters::Vector{Bool}  # update coder cluster centroids
    cluster_mass::Vector{Vector{R}}  # initial coder cluster mass (bias)
    clustering::Vector{String}  # Clustering algorithms for layers
    coder::Vector{Int}          # Coder options: 1 ⇒ 1D Kmeans;  2 ⇒ [a,o] experimental
    d::Vector{Int}              # Dimensions of layers clustering vector spaces
    decode_fast::Vector{Bool}   # Decode a (long) layer morpheme using Layer.sequences memory
    decode_num::Vector{Int}     # Length(s) of the down_plan(s) for the lower layer
    decode_slow::Vector{Bool}   # Decode layer morphemes as cartesian products of symbols
    discount::Vector{Vector{R}} # Value learning discounts for layer modules
    fitness::Vector{Int}        # Plans sorting criterion: 2 - by value, 3 - by probability assesment
    episodic_threshold::Vector{Int}     # Episodic memory size to start a new layer
    global_value_update::Vector{Bool}   # Choose global vs local morpheme value update
    initial_value::R            # Initial value of morpheme upon creation
    input_weights::Vector{F}    # Input feature weights for scaling to vectors
    internal_reward::Vector{R}  # Internal rewards for layers
    layers::Vector{Layer}
    learning_rate::Vector{Vector{Float}}  # α - value update rates, unique for each head
    l::Vector{Int}              # Max morpheme lengths in layers  #20521 ex m
    min_plan_length             # Minimal plan l ⇐ "Planning 0.5" lₚₗₐₙ
    planner_order::Vector{Int}  # 0: use the last parser stack symbol, >0: use more...
    planning_fast::Vector{Bool}
    planning_slow::Vector{Bool}
    plans_max::Vector{Vector{Int}}  # max number of plans chosen from the head R matrix
    plans_min::Vector{Int}      # Min number of plans valid for head voting results
    ppmi_threshold::Vector{Vector{R}}   # "Positive" PMI threshold (default zero)
    quorum::Vector{Float}       # Head voting quorum - majority share. Rounded to int ∈ 2:Layer.N
    thompson_sampling::Vector{String}   # Sampling algorithm : "Poisson", "Gauss", TBD...
    sampling_dispersion::Vector{R}      # Gaussian sampling dispersion
    scaler::Vector{F}           # Normalize input signals to ≈±1. (multiplier)
    rescaler::Vector{F}         # Restore input signals from encoded vectors (multiplier)
    semantic_threshold::Vector{Vector{R}} # R matrix value threshold for semantic memory growth
    step_reward::Vector{R}      # rewards > 0 / penalties < 0 for each layer
    sync_output::Vector{Bool}   # Update response plans to fit current time tick (cut the past)
    threading::Vector{Bool}     # Use multi-threading (layer-wise)
    update_rewards::Vector{Bool}
    update_values::Vector{Bool}
    validate_plans::Vector{Bool}# check plans to fit episodic memory before passing to the lower layer
    value_threshold::Vector{R}  # Min value to account for
    values::Vector{Vector{Int}} # Value-based choice criteria for layer modules
    winner_pair_count::Vector{R}# Head.R ⇑ for the winning pair of morphems in the parser stack
    working_memory::Vector{Int} # Parser memory value backpropagation range
    app::Dict                   # application parameters
    opt::Dict                   # development options
end
Adam() = Adam(  1, 1, [], [], [[]], [], [], "", [], [[]], [], [], [],
                Bool[], [], Bool[], [[]], [], [], [], R(0.), [], [], [], [[]], [], [], [],
                Bool[], Bool[], [], [], [[]], [], [], [], [], [],
                [[]], [], [], [], Bool[], Bool[], [], [], [[]], [], [], Dict(), Dict()
              )
