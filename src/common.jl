using Fed.Config: BaseConfig, VanillaConfig, QuantizedConfig,
    QDiffConfig, QDiffStaticConfig, GDConfig, GDStaticConfig
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

const base_config = BaseConfig(
    SERVERURL,
    NUM_COMM_ROUNDS,
    FRACTION_CLIENTS,
    NUM_TOTAL_CLIENTS
)

newVanillaConfig() = VanillaConfig(base_config)
newQuantizedConfig(::Type{T}) where T <: Unsigned = QuantizedConfig{T}(base_config)
newQDiffConfig(::Type{T}, is_client::Bool) where T <: Unsigned =
    QDiffConfig{T}(base_config, 256, is_client)

newQDiffStaticConfig(::Type{T}, is_client::Bool) where T <: Unsigned =
    QDiffStaticConfig{T}(base_config, 256, is_client)

newGDConfig(::Type{T}, port::Int, is_client::Bool) where T <: Unsigned =
    GDConfig{T}(
        base_config,
        256,
        sha1,
        # hash_crc32,
        round(T, 0.125 * sizeof(T) * 8), # % of each weight which goes in the basis
        "127.0.0.1",
        port,
        is_client
    )

newGDStaticConfig(::Type{T}, port::Int, is_client::Bool) where T <: Unsigned =
    GDStaticConfig{T}(
        base_config,
        256,
        # sha1,
        hash_crc32,
        round(T, 0.6 * sizeof(T) * 8), # % of each weight which goes in the basis
        "127.0.0.1",
        port,
        is_client
    )


"""
    newconfig()

Returns a new current configuration.
"""
# newconfig(_, _) = newVanillaConfig()
# newconfig(_, _) = newQuantizedConfig(UInt8)
# newconfig(_, is_client) = newQDiffConfig(UInt8, is_client)
# newconfig(_, is_client) = newQDiffStaticConfig(UInt16, is_client)
newconfig(port, is_client) = newGDConfig(UInt8, port, is_client)
# newconfig(port, is_client) = newGDStaticConfig(UInt8, port, is_client)

"""
    newmodel()

Returns a new current model.
"""
newmodel() = initialize_mlp(28*28, 200, 10)
