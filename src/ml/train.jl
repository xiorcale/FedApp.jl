using Flux
using Flux: gradient, logitcrossentropy, params


"""
    fit(client, weights)

Train the clients with the given weigths and return the updated weights.
"""
function fit(client::Client, weights::Vector{Float32})::Vector{Float32}
    @info "fit on client..."
    load_flatten_model!(client.model, weights)

    loss(x, y) = logitcrossentropy(client.model(x), y)
    epoch_loss = Vector{Float32}(undef, client.epochs)
   
    @inbounds for i = 1:client.epochs
        local train_loss = 0.0f0

        for batch in client.dataset
            grads = gradient(params(client.model)) do
                train_loss += loss(batch...)
            end
            Flux.update!(client.optimizer, params(client.model), grads)
        end

        epoch_loss[i] = train_loss / length(client.dataset)
    end

    loss_total = sum(epoch_loss) / length(epoch_loss)
    @info "loss = $loss_total"

    return flatten_model(client.model)
end