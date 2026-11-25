
import csv
import os
import re
import statistics
import subprocess

IMAGE = r"images\original\image3.ppm"
OUTPUT_DIR = "output"
EXECUTABLE = "main.exe"
RUNS = 5

# name, mode, MPI processes, OpenMP threads
CONFIGURATIONS = [
    ("Sequential", "seq", 1, 1),
    ("OpenMP_12", "omp", 1, 12),
    ("MPI_2", "mpi", 2, 1),
    ("MPI_4", "mpi", 4, 1),
    ("Hybrid_2x6", "hybrid", 2, 6),
    ("Hybrid_3x4", "hybrid", 3, 4),
    ("Hybrid_4x3", "hybrid", 4, 3),
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def execute(name, mode, processes, threads):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)

    output = os.path.join(OUTPUT_DIR, f"benchmark_{name}.ppm")

    if mode in ("mpi", "hybrid"):
        command = [
            "mpiexec", "-n", str(processes),
            EXECUTABLE, mode, IMAGE, output
        ]
    else:
        command = [EXECUTABLE, mode, IMAGE, output]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=env,
        check=True
    )

    match = re.search(
        r"Execution Time:\s*([\d.]+)\s*seconds",
        result.stdout + result.stderr
    )

    if not match:
        raise RuntimeError(
            f"Could not find execution time for {name}.\n"
            f"Output:\n{result.stdout}\n{result.stderr}"
        )

    return float(match.group(1))


results = []

print("Starting Gaussian Blur benchmark...")
print(f"Image: {IMAGE}")
print(f"Measured runs per configuration: {RUNS}\n")

for name, mode, processes, threads in CONFIGURATIONS:
    print(f"Testing {name} ({processes} MPI x {threads} threads)...")

    # One warm-up run, excluded from measurements
    execute(name, mode, processes, threads)

    times = []

    for run in range(1, RUNS + 1):
        elapsed = execute(name, mode, processes, threads)
        times.append(elapsed)
        print(f"  Run {run}: {elapsed:.6f} seconds")

    results.append({
        "name": name,
        "mode": mode,
        "processes": processes,
        "threads": threads,
        "total_workers": processes * threads,
        "times": times,
        "average": statistics.mean(times),
        "minimum": min(times),
        "maximum": max(times),
    })

baseline = next(
    result["average"]
    for result in results
    if result["mode"] == "seq"
)

csv_file = os.path.join(OUTPUT_DIR, "benchmark_results.csv")

with open(csv_file, "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "Configuration",
        "MPI Processes",
        "OpenMP Threads",
        "Total Workers",
        "Run 1 (s)",
        "Run 2 (s)",
        "Run 3 (s)",
        "Run 4 (s)",
        "Run 5 (s)",
        "Average (s)",
        "Minimum (s)",
        "Maximum (s)",
        "Speedup",
        "Efficiency (%)",
    ])

    for result in results:
        speedup = baseline / result["average"]
        efficiency = speedup / result["total_workers"] * 100

        writer.writerow([
            result["name"],
            result["processes"],
            result["threads"],
            result["total_workers"],
            *[f"{t:.6f}" for t in result["times"]],
            f"{result['average']:.6f}",
            f"{result['minimum']:.6f}",
            f"{result['maximum']:.6f}",
            f"{speedup:.3f}",
            f"{efficiency:.2f}",
        ])

print("\n========== BENCHMARK SUMMARY ==========")
print(f"{'Configuration':<18} {'Average (s)':>12} {'Speedup':>10} {'Efficiency':>12}")

for result in results:
    speedup = baseline / result["average"]
    efficiency = speedup / result["total_workers"] * 100

    print(
        f"{result['name']:<18} "
        f"{result['average']:>12.6f} "
        f"{speedup:>9.2f}x "
        f"{efficiency:>10.2f}%"
    )

print(f"\nCSV saved to: {csv_file}")
