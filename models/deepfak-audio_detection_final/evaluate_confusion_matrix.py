import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


MODEL_ID = "local/deepfak-audio_detection_final"
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
LABEL_ORDER = ["fake", "real"]


def add_backend_to_path() -> Path:
    current_file = Path(__file__).resolve()
    repo_root = current_file.parents[3]
    backend_dir = repo_root / "backend"
    sys.path.insert(0, str(backend_dir))
    return repo_root


def infer_label(file_path: Path) -> str | None:
    for part in file_path.parts:
        cleaned = part.strip().lower()
        if cleaned in {"fake", "spoof", "synthetic", "ai"}:
            return "fake"
        if cleaned in {"real", "bonafide", "genuine", "human"}:
            return "real"
    return None


def list_audio_files(dataset_dir: Path) -> list[Path]:
    files = []
    for path in dataset_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return sorted(files)


def compute_class_metrics(matrix: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
    metrics = {}
    for label in LABEL_ORDER:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in LABEL_ORDER if other != label)
        fn = sum(matrix[label][other] for other in LABEL_ORDER if other != label)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        metrics[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    return metrics


def evaluate(dataset_dir: Path, output_path: Path | None = None) -> dict:
    repo_root = add_backend_to_path()
    from deepfake_detector import DeepfakeVoiceDetector  # pylint: disable=import-error

    detector = DeepfakeVoiceDetector()
    detector.load(MODEL_ID)

    audio_files = list_audio_files(dataset_dir)
    if not audio_files:
        raise ValueError(f"No audio files found in: {dataset_dir}")

    matrix = {actual: {predicted: 0 for predicted in LABEL_ORDER} for actual in LABEL_ORDER}
    results = []
    skipped = []
    correct = 0

    for index, audio_path in enumerate(audio_files, start=1):
        actual = infer_label(audio_path)
        if actual not in LABEL_ORDER:
            skipped.append({"file": str(audio_path), "reason": "label_not_found_from_path"})
            continue

        try:
            analysis = detector.analyze(str(audio_path), model_id=MODEL_ID)
            predicted = analysis["prediction"].strip().lower()
            if predicted not in LABEL_ORDER:
                raise ValueError(f"Unexpected predicted label: {predicted}")
        except Exception as exc:  # noqa: BLE001
            skipped.append({"file": str(audio_path), "reason": str(exc)})
            continue

        matrix[actual][predicted] += 1
        is_correct = actual == predicted
        if is_correct:
            correct += 1

        results.append(
            {
                "file": str(audio_path.relative_to(repo_root)),
                "actual": actual,
                "predicted": predicted,
                "confidence": analysis["confidence"],
                "probabilities": analysis["probabilities"],
                "correct": is_correct,
            }
        )

        if index % 10 == 0 or index == len(audio_files):
            print(f"Processed {index}/{len(audio_files)} files...")

    evaluated_count = len(results)
    accuracy = correct / evaluated_count if evaluated_count else 0.0
    class_metrics = compute_class_metrics(matrix)

    report = {
        "model_id": MODEL_ID,
        "dataset_dir": str(dataset_dir),
        "evaluated_files": evaluated_count,
        "skipped_files": len(skipped),
        "accuracy": round(accuracy, 4),
        "confusion_matrix": {
            "labels": LABEL_ORDER,
            "rows_are_actual": True,
            "columns_are_predicted": True,
            "values": [[matrix[actual][predicted] for predicted in LABEL_ORDER] for actual in LABEL_ORDER],
        },
        "class_metrics": class_metrics,
        "mistakes": [item for item in results if not item["correct"]],
        "skipped": skipped,
    }

    if output_path:
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def print_report(report: dict) -> None:
    labels = report["confusion_matrix"]["labels"]
    values = report["confusion_matrix"]["values"]

    print("\n=== Evaluation Summary ===")
    print(f"Model: {report['model_id']}")
    print(f"Dataset: {report['dataset_dir']}")
    print(f"Evaluated files: {report['evaluated_files']}")
    print(f"Skipped files: {report['skipped_files']}")
    print(f"Accuracy: {report['accuracy']:.4f}")

    print("\n=== Confusion Matrix ===")
    print("actual \\ predicted | " + " | ".join(f"{label:>5}" for label in labels))
    print("-" * 38)
    for actual, row in zip(labels, values):
        row_text = " | ".join(f"{value:>5}" for value in row)
        print(f"{actual:>18} | {row_text}")

    print("\n=== Class Metrics ===")
    for label, metrics in report["class_metrics"].items():
        print(
            f"{label:>5} | precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f}"
        )

    if report["mistakes"]:
        print("\n=== Mistakes ===")
        for item in report["mistakes"][:20]:
            print(
                f"{item['file']} | actual={item['actual']} "
                f"predicted={item['predicted']} confidence={item['confidence']}"
            )


def save_confusion_matrix_plot(report: dict, image_path: Path) -> None:
    labels = report["confusion_matrix"]["labels"]
    values = report["confusion_matrix"]["values"]
    accuracy = report["accuracy"]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    image = ax.imshow(values, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(labels)), labels=[label.upper() for label in labels])
    ax.set_yticks(range(len(labels)), labels=[label.upper() for label in labels])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title(f"Confusion Matrix\nAccuracy = {accuracy:.4f}")

    threshold = max(max(row) for row in values) / 2 if values else 0
    for row_index, row in enumerate(values):
        for col_index, value in enumerate(row):
            color = "white" if value > threshold else "black"
            ax.text(col_index, row_index, str(value), ha="center", va="center", color=color, fontsize=13)

    fig.tight_layout()
    fig.savefig(image_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    default_dataset = repo_root / "dist" / "deepvoice_segments_50"
    default_output = Path(__file__).resolve().with_name("evaluation_report.json")
    default_image = Path(__file__).resolve().with_name("confusion_matrix.png")

    parser = argparse.ArgumentParser(description="Evaluate the local deepfake audio model.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=default_dataset,
        help="Dataset root directory. Labels are inferred from folder names such as FAKE/REAL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Optional JSON file to save the evaluation report.",
    )
    parser.add_argument(
        "--image-output",
        type=Path,
        default=default_image,
        help="PNG file to save the confusion matrix plot.",
    )
    args = parser.parse_args()

    report = evaluate(args.dataset_dir.resolve(), args.output.resolve() if args.output else None)
    print_report(report)
    if args.image_output:
        save_confusion_matrix_plot(report, args.image_output.resolve())
        print(f"Saved confusion matrix image to: {args.image_output.resolve()}")
    if args.output:
        print(f"\nSaved report to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
