using SHA
using CRC32c


"""
    hash_crc32(data)

Hash data with CRC32 and returns the hash as a `Vector{UInt8}`.
"""
function hash_crc32(data)
    hash = Vector{UInt8}(undef, 4)
    data = crc32c(data)
    hash[1] = (data & 0x000000ff)
    hash[2] = (data & 0x0000ff00) >> 8
    hash[3] = (data & 0x00ff0000) >> 16
    hash[4] = (data & 0xff000000) >> 24
    return hash
end


# config
const SERVERURL = "http://127.0.0.1:8080"
const NUM_COMM_ROUNDS = 100
const FRACTION_CLIENTS = 0.1f0
const NUM_TOTAL_CLIENTS = 100

newVanillaConfig() = VanillaConfig{Float32}(
    SERVERURL,
    NUM_COMM_ROUNDS,
    FRACTION_CLIENTS,
    NUM_TOTAL_CLIENTS
)

newQuantizedConfig(dtype::Type{T}, is_patcher::Bool) where T <: Unsigned = QuantizedConfig{dtype}(
    SERVERURL,
    NUM_COMM_ROUNDS,
    FRACTION_CLIENTS,
    NUM_TOTAL_CLIENTS,
    256,
    is_patcher
)

newGDConfig(dtype::Type{T}, is_patcher::Bool) where T <: Unsigned = GDConfig{dtype}(
    SERVERURL,
    NUM_COMM_ROUNDS,
    FRACTION_CLIENTS,
    NUM_TOTAL_CLIENTS,
    256,
    # sha1,
    hash_crc32,
    round(dtype, 0.3 * sizeof(dtype) * 8),  # 60% of each weight goes in the basis,
    is_patcher
)

"""
    newconfig()

Returns a new current configuration.
"""
# newconfig(_) = newVanillaConfig()
# newconfig(is_patcher) = newQuantizedConfig(UInt8, is_patcher)
newconfig(is_patcher) = newGDConfig(UInt8, is_patcher)

"""
    newmodel()

Returns a new current model. 
"""
newmodel() = initialize_mlp(28*28, 200, 10)