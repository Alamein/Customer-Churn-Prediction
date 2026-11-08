"""
End-to-end TrafficFlow pipeline: video → trajectories → features → models → report.

A single command chains the four CLIs in `src/`, dropping all artefacts under
one timestamped directory so the thesis-side write-up has a self-contained
results bundle per run.

Usage::

    # Full pipeline, dataset goes to runs/<timestamp>/
    uv run python main.py --video data/videos/input/sample.mp4

    # Custom output dir, cap to 600 frames, skip evaluation
    uv run python main.py --video sample.mp4 --out-dir runs/demo \\
        --max-frames 600 --skip-evaluate

    # Re-evaluate against a previously trained model directory
    uv run python main.py --video sample.mp4 --skip-track --skip-features \\
        --skip-train --out-dir runs/2026-05-16-1430
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import FeatureExtractor  # noqa: E402
from src.prediction.dataset import split_train_test  # noqa: E402
from src.prediction.evaluate import evaluate as evaluate_models  # noqa: E402
from src.prediction.train import ensure_label, train_lstm, train_rf  # noqa: E402
from src.tracking import VehicleTracker  # noqa: E402

logger = logging.getLogger("trafficflow.main")


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _git_sha() -> str | None:
    """Return the current commit SHA, or None if not a git repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _git_dirty() -> bool | None:
    """True if the working tree has uncommitted changes."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _library_versions() -> dict[str, str]:
    """Snapshot the versions of the libraries that materially affect results."""
    versions: dict[str, str] = {}
    for name in ("torch", "ultralytics", "sklearn", "xgboost", "numpy",
                 "pandas", "cv2"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[name] = "not installed"
    return versions


def _write_metadata(out_dir: Path, run_metadata: dict, pipeline_start: float) -> None:
    """Finalise and persist run_metadata.json so every run is auditable."""
    run_metadata["finished_at"] = datetime.now(UTC).isoformat()
    run_metadata["total_duration_sec"] = round(time.time() - pipeline_start, 2)
    meta_path = out_dir / "run_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(run_metadata, f, indent=2, default=str)
    logger.info("Wrote run metadata → %s", meta_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end TrafficFlow pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", required=True,
                        help="Input video — local path, http(s) URL, or rtsp:// stream.")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: runs/<timestamp>).")
    parser.add_argument("--config", default="config/model.yaml")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="YOLOv8 weights (auto-downloads if absent).")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Cap the tracking loop. Required for live RTSP/RTMP.")
    parser.add_argument("--time-window", type=float, default=60.0,
                        help="Feature aggregation window, seconds.")
    parser.add_argument("--grid-size", type=float, default=50.0,
                        help="Spatial grid cell size (px without homography).")
    parser.add_argument("--threshold-quantile", type=float, default=0.7,
                        help="Quantile for the proxy hotspot label.")
    parser.add_argument("--skip-track", action="store_true",
                        help="Reuse trajectories.csv from --out-dir.")
    parser.add_argument("--skip-features", action="store_true",
                        help="Reuse features.csv from --out-dir.")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip model training (and evaluation).")
    parser.add_argument("--skip-evaluate", action="store_true",
                        help="Skip evaluation only.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    out_dir = Path(args.out_dir or f"runs/{datetime.now():%Y%m%d-%H%M%S}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Pipeline output directory: %s", out_dir.resolve())

    traj_csv = out_dir / "trajectories.csv"
    features_csv = out_dir / "features.csv"
    models_dir = out_dir / "models"
    results_dir = out_dir / "results"

    # ---------------------------------------------------------- run metadata
    pipeline_start = time.time()
    run_metadata: dict = {
        "started_at": datetime.now(UTC).isoformat(),
        "args": vars(args),
        "git": {"sha": _git_sha(), "dirty": _git_dirty()},
        "platform": {
            "python": sys.version,
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "libraries": _library_versions(),
        "stages": {},
    }

    # ----------------------------------------------------------- 1) tracking
    if args.skip_track:
        if not traj_csv.exists():
            parser.error(f"--skip-track requires {traj_csv} to exist.")
        logger.info("[1/4] Skipping tracking — reusing %s", traj_csv)
        tracker_fps: float | None = None
        run_metadata["stages"]["tracking"] = {"skipped": True}
    else:
        logger.info("[1/4] Tracking %s …", args.video)
        t0 = time.time()
        tracker = VehicleTracker(
            model_path=args.model, confidence=args.confidence,
        )
        stats = tracker.process_video(
            video_path=args.video,
            output_csv=str(traj_csv),
            max_frames=args.max_frames,
        )
        tracker_fps = stats["fps"]
        run_metadata["stages"]["tracking"] = {
            "duration_sec": round(time.time() - t0, 2),
            **stats,
        }
        logger.info(
            "[1/4] Done — %d frames @ %.1f fps, %d unique vehicles, %d detections",
            stats["frames"], stats["fps"],
            stats["unique_vehicles"], stats["detections"],
        )

    # ----------------------------------------------------------- 2) features
    if args.skip_features:
        if not features_csv.exists():
            parser.error(f"--skip-features requires {features_csv} to exist.")
        logger.info("[2/4] Skipping feature extraction — reusing %s", features_csv)
        import pandas as pd
        features = pd.read_csv(features_csv)
        run_metadata["stages"]["features"] = {"skipped": True}
    else:
        # If we skipped tracking we need to recover fps somehow — fall back
        # to the YAML default rather than guessing, and tell the user.
        fps = tracker_fps if tracker_fps is not None else 30.0
        if args.skip_track:
            logger.warning(
                "fps unknown (tracking was skipped) — assuming %.1f fps. "
                "Override with the FPS embedded in your video metadata if needed.",
                fps,
            )
        logger.info("[2/4] Extracting features (fps=%.1f, window=%.0fs, grid=%.0fpx) …",
                    fps, args.time_window, args.grid_size)
        t0 = time.time()
        extractor = FeatureExtractor(
            fps=fps,
            time_window=args.time_window,
            spatial_grid_size=args.grid_size,
        )
        features = extractor.extract_from_csv(str(traj_csv))
        features.to_csv(features_csv, index=False)
        run_metadata["stages"]["features"] = {
            "duration_sec": round(time.time() - t0, 2),
            "rows": int(len(features)),
            "fps": fps,
            "time_window_sec": args.time_window,
            "grid_size_px": args.grid_size,
        }
        logger.info("[2/4] Wrote %d feature rows → %s", len(features), features_csv)

    if args.skip_train:
        logger.info("Skipping training (and evaluation). Done.")
        _write_metadata(out_dir, run_metadata, pipeline_start)
        return

    # ----------------------------------------------------------- 3) training
    logger.info("[3/4] Training RF + LSTM …")
    cfg = _load_yaml(args.config)
    df_labelled = ensure_label(features, args.threshold_quantile)
    test_size = cfg["training"]["test_size"]
    train_df, test_df = split_train_test(df_labelled, test_size=test_size)

    rf_path = models_dir / "rf_model.joblib"
    lstm_path = models_dir / "lstm_model.pt"

    t0 = time.time()
    rf_metrics = train_rf(train_df, test_df, cfg, str(rf_path))
    try:
        lstm_metrics = train_lstm(df_labelled, train_df, test_df, cfg, str(lstm_path))
    except ValueError as exc:
        logger.warning("LSTM training skipped: %s", exc)
        lstm_metrics = {"skipped_reason": str(exc)}

    training_report = {"random_forest": rf_metrics, "lstm": lstm_metrics}
    with open(out_dir / "training_report.json", "w") as f:
        json.dump(training_report, f, indent=2)
    run_metadata["stages"]["training"] = {
        "duration_sec": round(time.time() - t0, 2),
        **training_report,
    }
    logger.info(
        "[3/4] Training done — RF: acc=%.3f, AUC=%s | LSTM: %s",
        rf_metrics["accuracy"],
        f"{rf_metrics['auc_roc']:.3f}" if rf_metrics.get("auc_roc") is not None else "n/a",
        f"acc={lstm_metrics['accuracy']:.3f}, AUC={lstm_metrics['auc_roc']:.3f}"
        if "accuracy" in lstm_metrics else lstm_metrics.get("skipped_reason", "skipped"),
    )

    if args.skip_evaluate:
        logger.info("Skipping evaluation.")
        _write_metadata(out_dir, run_metadata, pipeline_start)
        return

    # ----------------------------------------------------------- 4) evaluation
    logger.info("[4/4] Evaluating + producing comparison plots …")
    t0 = time.time()
    eval_report = evaluate_models(
        df_labelled,
        config=cfg,
        out_dir=results_dir,
        rf_path=str(rf_path) if rf_path.exists() else None,
        lstm_path=str(lstm_path) if lstm_path.exists() else None,
        threshold_quantile=args.threshold_quantile,
    )
    run_metadata["stages"]["evaluation"] = {
        "duration_sec": round(time.time() - t0, 2),
        "report": eval_report,
    }
    logger.info("[4/4] Wrote evaluation artefacts → %s", results_dir)
    _write_metadata(out_dir, run_metadata, pipeline_start)

    # ------------------------------------------------------------- summary
    print("\n=== Pipeline complete ===")
    print(f"Output directory: {out_dir.resolve()}")
    print(f"  trajectories:    {traj_csv}")
    print(f"  features:        {features_csv}")
    print(f"  models:          {models_dir}/")
    print(f"  results:         {results_dir}/")
    print()
    if eval_report:
        print(f"{'Model':<15} {'Accuracy':>10} {'AUC-ROC':>10}")
        for name, m in eval_report.items():
            auc = m.get("auc_roc")
            auc_s = f"{auc:.3f}" if auc is not None else "  n/a "
            print(f"{name:<15} {m['accuracy']:>10.3f} {auc_s:>10}")


if __name__ == "__main__":
    main()
