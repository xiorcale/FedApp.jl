using Fed
using Flux

function start_server()
    # config
    host = "127.0.0.1"
    port = 8080
    weights = MODEL |> flatten_model
    strategy = Fed.Server.federated_averaging

    # instanciate the server
    central_node = Fed.Server.CentralNode(host, port)
    config = Fed.Server.Config(weights, strategy)

    # start the server
    @info "Server started on [http://$host:$port]"
    Fed.Server.start_server(central_node, config)
end