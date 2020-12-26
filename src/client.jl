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
        model(),
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
    host = "127.0.0.1"

    # config = VanillaConfig{Float32}(
    #     "http://127.0.0.1:8080", 
    #     NUM_COMM_ROUNDS, 
    #     FRACTION_CLIENTS, 
    #     NUM_TOTAL_CLIENTS
    # )
    # config = QuantizedConfig{UInt8}(
    #     "http://127.0.0.1:8080",
    #     NUM_COMM_ROUNDS,
    #     FRACTION_CLIENTS,
    #     NUM_TOTAL_CLIENTS,
    # )
    config = GDConfig{UInt8}(
        "http://127.0.0.1:8080",
        NUM_COMM_ROUNDS,
        FRACTION_CLIENTS,
        NUM_TOTAL_CLIENTS,
        256,
        sha1,
        0x05,
        "./permutations.jld"
    )

    node = Fed.Client.Node{config.common.dtype}(host, port, curry(fit, client), config)

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