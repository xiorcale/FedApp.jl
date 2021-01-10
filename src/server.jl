using Fed
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
    config = newconfig(9090, false)

    # config
    host = "127.0.0.1"
    port = 8080
    weights = flatten_model(server.model)
    strategy = Fed.Server.federated_averaging
    eval_hook = (weights::Vector{Float32}) -> evaluate(server, weights)
    
    central_node = Fed.Server.CentralNode(
        host, 
        port, 
        weights,
        strategy,
        eval_hook,
        config
    )

    # start the server
    @info "Server started on [http://$host:$port]"
    Fed.Server.start(central_node)
end
