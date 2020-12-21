using Fed
using Fed: curry
using Flux
using Flux.Data: DataLoader


struct Server
    model::Chain
    dataset::DataLoader

    Server() = new(MODEL, get_testdata(isflat=true))
end


function start_server()
    # config
    host = "127.0.0.1"
    port = 8080
    weights = MODEL |> flatten_model
    strategy = Fed.Server.federated_averaging

    # instanciate the server
    server = Server()
    central_node = Fed.Server.CentralNode(host, port, curry(evaluate, server))
    config = Fed.Server.Config(weights, strategy)

    # start the server
    @info "Server started on [http://$host:$port]"
    Fed.Server.start_server(central_node, config)
end
