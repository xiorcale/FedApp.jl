using Fed
using Fed: curry, VanillaConfig, QuantizedConfig, GDConfig
using Flux
using SHA

struct Server
    model::Chain
    dataset::DataLoader

    Server() = new(
        model(),
        get_testdata(isflat=true)
    )
end


function start_server()
    server = Server()
    num_comm_rounds = 100
    
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
    #     NUM_TOTAL_CLIENTS
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

    # config
    host = "127.0.0.1"
    port = 8080
    weights = flatten_model(server.model)
    strategy = Fed.Server.federated_averaging
    eval_hook = curry(evaluate, server)
    
    central_node = Fed.Server.CentralNode{config.common.dtype}(
        host, 
        port, 
        weights,
        strategy,
        eval_hook,
        config
    )

    # start the server
    @info "Server started on [http://$host:$port]"
    Fed.Server.start_server(central_node)
end
