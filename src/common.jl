using SHA
using CRC32c

# model used for training
model() = initialize_mlp(28*28, 200, 10)


function initialize_config()
    # endpoints
    serverurl = "http://127.0.0.1:8080"
    register_node = "/register"
    fit_node = "/fit"
    gd_bases = "/bases"

    # quantization
    # qdtype = Float32
    # qmin = -1.0
    # qmax = 1.0

    qdtype = UInt8
    qmin = 0x00
    qmax = 0xFF

    # qdtype = UInt16
    # qmin = 0x0000
    # qmax = 0xFFFF


    # gd
    chunksize = 256
    fingerprint = sha1
    # fingerprint = crc32c
    permutations_file = "./permutations.jld"

    # transform
    msbsize = 0x05
    # msbsize = 0x000B

    # serialization
    # payload_serde = VanillaPayloadSerde()
    # payload_serde =  QuantizedPayloadSerde{qdtype}(qmin, qmax)
    payload_serde = GDPayloadSerde{qdtype}(qmin, qmax, chunksize, fingerprint, msbsize, permutations_file)

    return Fed.Server.Config{qdtype}(
        serverurl,
        register_node,
        fit_node,
        gd_bases,
        qdtype,
        qmin,
        qmax,
        chunksize,
        fingerprint,
        permutations_file,
        msbsize,
        payload_serde
    )
end
