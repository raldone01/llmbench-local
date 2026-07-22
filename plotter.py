import os
import re
import matplotlib.pyplot as plt
import numpy as np

INPUT_DIR = "ollama_results"
PLOT_DIR = "plots"


def parse_duration_to_seconds(time_str):
    time_str = time_str.strip()
    seconds = 0.0

    if "ms" in time_str:
        ms_part = re.search(r"([\d\.]+)ms", time_str)
        if ms_part:
            seconds += float(ms_part.group(1)) / 1000.0
            time_str = re.sub(r"[\d\.]+ms", "", time_str)

    if "m" in time_str:
        m_part = re.search(r"([\d\.]+)m", time_str)
        if m_part:
            seconds += float(m_part.group(1)) * 60.0
            time_str = re.sub(r"[\d\.]+m", "", time_str)

    if "s" in time_str:
        s_part = re.search(r"([\d\.]+)s", time_str)
        if s_part:
            seconds += float(s_part.group(1))

    return seconds


def parse_files():
    # Structure: host_data["hostname"]["version"]["model_name"]["metric"] = value
    host_data = {}

    if not os.path.exists(INPUT_DIR):
        print(f"Directory {INPUT_DIR} not found. Run runner.py first.")
        return host_data

    # Iterate through hostname directories
    for hostname in os.listdir(INPUT_DIR):
        host_dir = os.path.join(INPUT_DIR, hostname)
        if not os.path.isdir(host_dir):
            continue

        host_data[hostname] = {}

        # Iterate through version directories
        for version in os.listdir(host_dir):
            version_dir = os.path.join(host_dir, version)
            if not os.path.isdir(version_dir):
                continue

            data = {}
            for filename in os.listdir(version_dir):
                if not filename.endswith("_stats.txt"):
                    continue

                filepath = os.path.join(version_dir, filename)
                model_data = {}
                model_name = "Unknown"

                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("Model:"):
                            model_name = line.split("Model:")[1].strip()

                        if (
                            ":" in line
                            and "duration" in line
                            and "total" not in line.lower()[:5]
                        ):
                            key, val = line.split(":", 1)
                            if "duration" in key:
                                model_data[key.strip()] = parse_duration_to_seconds(val)
                        elif "token(s)" in line:
                            key, val = line.split(":", 1)
                            model_data[key.strip()] = float(re.sub(r"[^\d\.]", "", val))
                        elif "tokens/s" in line:
                            key, val = line.split(":", 1)
                            model_data[key.strip()] = float(re.sub(r"[^\d\.]", "", val))

                if model_data:
                    if (
                        "eval count" in model_data
                        and "eval duration" in model_data
                        and "eval rate" not in model_data
                    ):
                        model_data["eval rate"] = (
                            model_data["eval count"] / model_data["eval duration"]
                        )
                    data[model_name] = model_data

            if data:
                host_data[hostname][version] = data

    return host_data


def plot_data(host_data):
    if not host_data:
        print("No data found to plot.")
        return

    plt.style.use("ggplot")

    for hostname, versions in host_data.items():
        for version, data in versions.items():
            print(f"Generating plots for host: {hostname} (Ollama v{version})...")

            target_plot_dir = os.path.join(PLOT_DIR, hostname, version)
            if not os.path.exists(target_plot_dir):
                os.makedirs(target_plot_dir)

            models = list(data.keys())
            x = np.arange(len(models))
            width = 0.25

            # --- PLOT 1: Durations ---
            fig, ax = plt.subplots(figsize=(12, 7))
            load_durs = [data[m].get("load duration", 0) for m in models]
            prompt_durs = [data[m].get("prompt eval duration", 0) for m in models]
            eval_durs = [data[m].get("eval duration", 0) for m in models]

            ax.bar(x - width, load_durs, width, label="Load Duration (s)")
            ax.bar(x, prompt_durs, width, label="Prompt Eval (s)")
            ax.bar(x + width, eval_durs, width, label="Eval Duration (s)")

            ax.set_ylabel("Seconds")
            ax.set_title(f"Time Spent per Phase - {hostname} (v{version})")
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha="right")
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(target_plot_dir, "durations_comparison.png"))
            plt.close()

            # --- PLOT 2: Token Generation Speeds ---
            fig, ax = plt.subplots(figsize=(12, 7))
            prompt_rates = [data[m].get("prompt eval rate", 0) for m in models]
            eval_rates = [data[m].get("eval rate", 0) for m in models]

            width_2 = 0.35
            ax.bar(
                x - width_2 / 2,
                prompt_rates,
                width_2,
                label="Prompt Eval Rate (tokens/s)",
            )
            ax.bar(
                x + width_2 / 2, eval_rates, width_2, label="Generation Rate (tokens/s)"
            )

            ax.set_ylabel("Tokens per Second")
            ax.set_title(f"Processing Speed Comparison - {hostname} (v{version})")
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha="right")
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(target_plot_dir, "token_rates_comparison.png"))
            plt.close()

            # --- PLOT 3: Output Token Counts ---
            fig, ax = plt.subplots(figsize=(12, 7))
            eval_counts = [data[m].get("eval count", 0) for m in models]

            ax.bar(x, eval_counts, width * 2, color="teal")
            ax.set_ylabel("Generated Tokens")
            ax.set_title(f"Amount of Tokens Generated - {hostname} (v{version})")
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(target_plot_dir, "token_counts_comparison.png"))
            plt.close()

    print(f"All plots successfully generated and saved inside '{PLOT_DIR}/'.")


if __name__ == "__main__":
    extracted_data = parse_files()
    plot_data(extracted_data)
