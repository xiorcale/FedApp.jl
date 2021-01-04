using Fed
using Fed: curry, VanillaConfig, QuantizedConfig, GDConfig
using Flux
using SHA

struct Server
    model::Chain
    dataset::DataLoader

    Server() = new(
        newmodel(),
        get_testdata(isflat=true)
    )
end


function start_server()
    server = Server()
    config = newconfig(false)

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
