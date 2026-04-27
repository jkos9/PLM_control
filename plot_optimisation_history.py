import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_history(csv_path: Path, metric: str):
	iterations = []
	values = []
	algo_name = ""

	with csv_path.open("r", encoding="utf-8", newline="") as handle:
		reader = csv.DictReader(handle)
		for row in reader:
			if not algo_name:
				algo_name = (row.get("optimizer_algorithm") or "").strip()

			iter_raw = row.get("iteration")
			metric_raw = row.get(metric)
			if not iter_raw or not metric_raw:
				continue

			try:
				iterations.append(int(float(iter_raw)))
				values.append(float(metric_raw))
			except ValueError:
				continue

	if not iterations:
		raise ValueError(f"No valid points found in {csv_path.name} for metric '{metric}'.")

	return (algo_name or csv_path.stem), iterations, values


def get_default_files(history_dir: Path):

	preferred = [
		"zernike_optimization_history_aspgd_1_m1-8_p_0.5_lr_0.2_.csv",
		"zernike_optimization_history_nm_1_m1-8_p_0.4.csv",
		"zernike_optimization_history_shc_1_m1-8_p_0.4.csv",
	]

	
	#preferred = [
	#	"zernike_optimization_history_aspgd_m1-15_p_0.4.csv",
	#	"zernike_optimization_history_nm_m1-15-p_0.4.csv",
	#	"zernike_optimization_history_shc_1_m1-15_p_0.4.csv",
	#]


	preferred_paths = [history_dir / name for name in preferred if (history_dir / name).exists()]
	if preferred_paths:
		return preferred_paths

	return sorted(history_dir.glob("zernike_optimization_history_*.csv"))


def parse_args():
	parser = argparse.ArgumentParser(
		description="Plot XT evolution versus iteration for optimizer history CSV files."
	)
	parser.add_argument(
		"files",
		nargs="*",
		type=Path,
		help="Optional CSV paths. If omitted, defaults in zernike_optimization are used.",
	)
	parser.add_argument(
		"--history-dir",
		type=Path,
		default=Path("zernike_optimization"),
		help="Directory that contains history CSV files.",
	)
	parser.add_argument(
		"--metric",
		type=str,
		default="xt_db",
		help="CSV column to plot on y-axis (default: xt_db).",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=None,
		help="Optional output image path. If omitted, the plot is shown interactively.",
	)
	return parser.parse_args()


def main():
	args = parse_args()
	root_dir = Path(__file__).resolve().parent
	history_dir = args.history_dir if args.history_dir.is_absolute() else (root_dir / args.history_dir)

	if args.files:
		csv_files = [path if path.is_absolute() else (root_dir / path) for path in args.files]
	else:
		csv_files = get_default_files(history_dir)

	if not csv_files:
		raise SystemExit(f"No CSV files found in {history_dir}")

	plt.figure(figsize=(7, 4))
	used_labels = set()

	for csv_file in csv_files:
		label, iterations, values = load_history(csv_file, args.metric)

		if label in used_labels:
			label = f"{label} ({csv_file.stem})"
		used_labels.add(label)

		plt.plot(iterations, values, marker="o", markersize=3, linewidth=1.4, label=label)

	plt.title("Mode-group XT evolution across optimization algorithms for first 8 modes")
	plt.xlabel("Iteration")
	plt.ylabel(f"{args.metric} (dB)")
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.savefig(root_dir / "xt_optimization_history_first_8_modes.png", dpi=200)

	if args.output is not None:
		output_path = args.output if args.output.is_absolute() else (root_dir / args.output)
		output_path.parent.mkdir(parents=True, exist_ok=True)
		plt.savefig(output_path, dpi=200)
		print(f"Saved plot to {output_path}")
	else:
		plt.show()


if __name__ == "__main__":
	main()





