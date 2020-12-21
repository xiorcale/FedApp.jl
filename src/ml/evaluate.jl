using Flux
using Flux: logitcrossentropy, onecold
using Flux.Data: DataLoader


"""
    compute_loss(model, dataset)

Compute the loss of `model` on `dataset`.
"""
function compute_loss(model::Chain, dataset::DataLoader)::Float32
    loss = 0.0f0
    for (x, y) in dataset
        loss += logitcrossentropy(model(x), y)
    end
    loss / length(dataset)
end


"""
    compute_accuracy(model, dataset)

Compute the accuracy of `model` on `dataset`.
"""
function compute_accuracy(model::Chain, dataset::DataLoader)::Float32
    acc = 0.0f0
    for (x, y) in dataset
        acc += sum(onecold(model(x)) .== onecold(y)) * 1 / size(x, 2)
    end
    acc / length(dataset)
end

"""
    evaluate(server, weights)

Evaluates the `server` model with the given weights on the `server` test set.
Returns a tuple (loss, accuracy).
"""
function evaluate(server::Server, weights::Vector{Float32})::Tuple{Float32, Float32}
    load_flatten_model!(server.model, weights)

    loss = compute_loss(server.model, server.dataset)
    acc = compute_accuracy(server.model, server.dataset)

    return loss, acc
end
