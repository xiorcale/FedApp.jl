module FedApp

include("ml/model.jl")
include("ml/mnist.jl")

include("common.jl")

include("server.jl")
include("client.jl")

include("ml/train.jl")
include("ml/evaluate.jl")

end # module
