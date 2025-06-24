#!/bin/bash

# Throughput benchmark script for SprayList
# Similar to https://github.com/jkopinsky/SprayList/blob/master/run_throughput.sh

echo "SprayList Throughput Benchmarks"
echo "==============================="
echo ""

# Build the project first
echo "Building project..."
cargo build --release --bench spraylist_bench

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo ""
echo "Running throughput scaling benchmark..."
echo "Thread counts: 1, 2, 4, 8, 16"
echo ""

# Run the throughput scaling benchmark
cargo bench --bench spraylist_bench -- bench_throughput_scaling --verbose

echo ""
echo "Running mixed workload benchmarks..."
echo "Update percentages: 0%, 25%, 50%, 75%, 100%"
echo ""

# Run mixed workload benchmarks
cargo bench --bench spraylist_bench -- bench_mixed_workloads --verbose

echo ""
echo "Running parameter comparison benchmarks..."
echo ""

# Run parameter benchmarks
cargo bench --bench spraylist_bench -- bench_spray_parameters --verbose

echo ""
echo "Benchmarks complete!"
echo ""
echo "For detailed results, check target/criterion/*/report/index.html"
echo "Raw data is available in target/criterion/"