using SHA
using CRC32c

# model used for training
model() = initialize_mlp(28*28, 200, 10)

const NUM_COMM_ROUNDS = 100
const FRACTION_CLIENTS = 0.1f0
const NUM_TOTAL_CLIENTS = 100
