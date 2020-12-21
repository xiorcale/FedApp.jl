using Flux


#-----------
# Tools
#-----------

"""
    flatten_model(model)

Flattens the model weights into a 1D array.
"""
flatten_model(model::Chain) = reduce(vcat, [layer[:] for layer in Flux.params(model)])


"""
    load_flatten_model!(model, flatten_model)

Reshapes the `flatten_model` and load it into `model`.
"""
function load_flatten_model!(model::Chain, flatten_model::Vector{Float32})
    weights = [similar(layer) for layer in Flux.params(model)]

    start = 1
    for i in 1:length(weights)
        shape = size(weights[i])
        stop = reduce(*, shape)
        weights[i] = reshape(flatten_model[start:(start - 1 + stop)], shape)
        start += stop
    end

    Flux.loadparams!(model, weights)
end


# ---------
# MLP
# ---------

"""
    initialize_mlp(input, hidden, output)

Create a 2-hidden layers MLP of hidden params each.
"""
function initialize_mlp(input::Int, hidden::Int, output::Int)
    return Chain(
        Dense(input, hidden, relu),
        Dense(hidden, hidden, relu),
        Dense(hidden, output),
    )
end


# ---------
# CNN
# ---------

"""
    initialize_cnn(imgsize = (28,28,1), numclasses=10)

Create a 2-conv layers CNN to work with the MNIST dataset.
"""
function initialize_cnn(imgsize = (28,28,1), numclasses::Int=10)
    return Chain(
        # 5x5x1x32 + 32 = 832
        Conv((5, 5), imgsize[3]=>32, pad=(1,1), relu),
        MaxPool((2,2)),

        # 5x5x32x64 + 64 = 51264
        Conv((5, 5), 32=>64, pad=(1,1), relu),
        MaxPool((2,2)),

        flatten,

        # 5*5*64 * 512 + 512 = 819712
        Dense(1600, 512, relu),

        # 512 * 10 + 10 = 5130
        Dense(512, numclasses),
    )
end
