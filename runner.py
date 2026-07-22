import subprocess
import platform
import psutil
import re
import os

MODELS = [
    "starcoder2:3b     ",
    "starcoder2:7b     ",
    "starcoder2:15b    ",
    "qwen2.5-coder:7b  ",
    "qwen2.5-coder:14b ",
    "qwen3-coder:30b   ",
    "qwen3.6:latest    ",
    "gemma4:12b        ",
    "gemma4:26b        ",
    "gemma4:31b        ",
    "ornith:35b        ",
]

PROMPT = "Write a python program showing off the astar algorithm."
OUTPUT_DIR = "ollama_results"


def get_ollama_version():
    """Gets the installed ollama version."""
    try:
        output = subprocess.check_output(["ollama", "--version"], text=True).strip()
        # Extracts version string like '0.3.14' from 'ollama version is 0.3.14'
        match = re.search(r"\d+\.\d+\.\d+(-\w+)?", output)
        return match.group(0) if match else "unknown_version"
    except Exception:
        return "unknown_version"


def get_cpu_name():
    """OS-specific commands to get the actual human-readable CPU name."""
    os_name = platform.system()
    try:
        if os_name == "Windows":
            output = subprocess.check_output(
                ["wmic", "cpu", "get", "name"], text=True
            ).strip()
            lines = [line.strip() for line in output.split("\n") if line.strip()]
            if len(lines) > 1:
                return lines[1]
        elif os_name == "Darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        elif os_name == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


def get_system_stats():
    stats = []
    stats.append(f"Hostname: {platform.node()}")

    cpu_count = psutil.cpu_count(logical=True)
    cpu_type = get_cpu_name()
    stats.append(f"CPU: {cpu_count} cores, Type: {cpu_type}")

    ram = psutil.virtual_memory()
    total_ram = ram.total / (1024**3)
    avail_ram = ram.available / (1024**3)
    stats.append(f"RAM: {avail_ram:.2f} GB available / {total_ram:.2f} GB total")

    try:
        gpu_info = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.free",
                    "--format=csv,noheader",
                ],
                text=True,
            )
            .strip()
            .split("\n")
        )
        for i, gpu in enumerate(gpu_info):
            stats.append(f"GPU {i}: {gpu}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        stats.append("GPU: No NVIDIA GPUs found or nvidia-smi not in PATH.")

    return "\n".join(stats)


def run_ollama_and_save():
    # Sanitize hostname and version for folder creation
    raw_hostname = platform.node()
    safe_hostname = "".join(c if c.isalnum() else "_" for c in raw_hostname)

    raw_version = get_ollama_version()
    safe_version = "".join(c if c.isalnum() or c in ".-" else "_" for c in raw_version)

    target_dir = os.path.join(OUTPUT_DIR, safe_hostname, safe_version)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    sys_stats = get_system_stats()
    print(f"--- System Stats ({raw_hostname} - Ollama v{raw_version}) ---")
    print(sys_stats)
    print("--------------------\n")

    metrics_to_track = [
        "total duration",
        "load duration",
        "prompt eval count",
        "prompt eval duration",
        "prompt eval rate",
        "eval count",
        "eval duration",
    ]

    for model in MODELS:
        safe_model_name = model.replace(":", "_")
        filepath = os.path.join(target_dir, f"{safe_model_name}_stats.txt")

        # Check if already tested
        if os.path.exists(filepath):
            print(f"\nSkipping {model}... (Stats file already exists at {filepath})")
            continue

        # Pull the model first
        print(f"\n[{model}] Pulling model...")
        subprocess.run(["ollama", "pull", model])

        # Run and evaluate the model
        print(f"[{model}] Running prompt...")
        command = ["ollama", "run", model, "--verbose", PROMPT]
        result = subprocess.run(command, capture_output=True, text=True)

        extracted_metrics = []
        for line in result.stderr.splitlines():
            line = line.strip()
            for metric in metrics_to_track:
                if line.startswith(metric + ":"):
                    extracted_metrics.append(line)

        if not extracted_metrics:
            print(f"  -> Failed to extract metrics for {model}.")
            continue

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Model: {model}\n")
            f.write(f"Ollama Version: {raw_version}\n")
            f.write("--- System Stats ---\n")
            f.write(sys_stats + "\n")
            f.write("--- Ollama Stats ---\n")
            for m in extracted_metrics:
                f.write(m + "\n")

        print(f"  -> Saved stats to {filepath}")


if __name__ == "__main__":
    run_ollama_and_save()
