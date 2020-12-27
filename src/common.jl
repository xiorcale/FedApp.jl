using SHA
using CRC32c

# config
const SERVERURL = "http://127.0.0.1:8080"
const NUM_COMM_ROUNDS = 100
const FRACTION_CLIENTS = 0.1f0
const NUM_TOTAL_CLIENTS = 100

newVanillaConfig() = VanillaConfig{Float32}(SERVERURL, NUM_COMM_ROUNDS, FRACTION_CLIENTS, NUM_TOTAL_CLIENTS)
newQuantizedConfig(dtype::Type{T}) where T <: Unsigned = QuantizedConfig{dtype}(SERVERURL, NUM_COMM_ROUNDS, FRACTION_CLIENTS, NUM_TOTAL_CLIENTS)
newGDConfig(dtype::Type{T}) where T <: Unsigned = GDConfig{dtype}(SERVERURL, NUM_COMM_ROUNDS, FRACTION_CLIENTS, NUM_TOTAL_CLIENTS, 256, sha1, 0x05, "./permutations.jld")

"""
    newconfig()

Returns a new current configuration.
"""
newconfig() = newGDConfig(UInt8)

"""
    newmodel()

Returns a new current model. 
"""
newmodel() = initialize_mlp(28*28, 200, 10)