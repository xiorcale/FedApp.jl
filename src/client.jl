using Fed
using Fed: curry, VanillaConfig, QuantizedConfig, GDConfig
using Flux
using Flux.Data: DataLoader
using Flux.Optimise: Momentum


struct Client
    model::Chain
    dataset::DataLoader
    optimizer::Momentum
    epochs::Int

    Client(dataset) = new(
        newmodel(),
        dataset,
        Momentum(0.01, 0.5),
        5 # epochs
    )
end


"""
    start_client(dataset, port)

Starts one client with the given dataset on the given localhost port.
"""
function start_client(dataset, port::Int)
    client = Client(dataset)
    config = newconfig(true)

    host = "127.0.0.1"
    train_hook = curry(fit, client)

    node = Fed.Client.Node{config.common.dtype}(host, port, train_hook, config)

    @info "Client started on [http://$host:$port]"
    Fed.Client.start_client(node)
end


"""
    start_client()

Starts 100 clients asynchronously adn return the tasks running them.
"""
function start_clients(num_clients::Int)
    num_sample = 60000
    step = Int(floor(num_sample / num_clients))

    tasks = [
        @async start_client(get_traindata((i * step + 1):((i+1) * step), isflat=true), 8081+i)
        for i in 0:num_clients-1
    ]
    
    return tasks
end


"""
    stop_clients(tasks)

Stops the given `tasks` which are running the clients.
"""
function stop_clients(tasks)
    for task in tasks
        @async Base.throwto(task, InterruptException())
    end
end