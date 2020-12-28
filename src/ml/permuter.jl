using Flux
using Flux: Chain, gradient, logitcrossentropy, params
using Flux.Data: DataLoader
using Flux.Optimise: Momentum
using JLD


"""
    generate_permutations(model, dataset, [optimizer])

Generate a JLD permutations file which shifts high-entropy weight at the end of
the array. To do so, a model and dataset are trained on a few eopchs to generate
the entropy pattern of the weights.
"""
function generate_permutations(model::Chain, dataset::DataLoader; optimizer=Momentum(0.01, 0.5))
    loss(x, y) = logitcrossentropy(model(x), y)

    num_epochs = 5
    weights = Vector{Vector{Float32}}(undef, num_epochs+1)
    weights[1] = flatten_model(model)

    for i = 2:num_epochs+1
        println("epochs [$(i-1)/$(num_epochs)]")
        local train_loss = 0.0f0
        for (i, batch) in enumerate(dataset)
            grads = gradient(params(model)) do
                train_loss += loss(batch...)
            end
            Flux.update!(optimizer, params(model), grads)
        end
        train_loss /= length(dataset)
        @show train_loss

        weights[i] = flatten_model(model)
    end

    weights_diff = sum([abs.(weights[i-1] - weights[i]) for i in 2:num_epochs+1]) / num_epochs
   
    perms = sortperm(weights_diff)

    JLD.save("permutations.jld", "permutations", perms)
end
