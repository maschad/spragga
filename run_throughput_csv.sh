#!/bin/bash

# CSV throughput benchmark script for SprayList
# Similar output format to the original C implementation

echo "Building throughput test binary..."
cargo build --release --bin throughput_test

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo ""
echo "Running CSV throughput scaling test..."
echo ""

# Run scaling test with CSV output
./target/release/throughput_test --scaling --csv --duration 3 --total-ops 1000000 --update-pct 100