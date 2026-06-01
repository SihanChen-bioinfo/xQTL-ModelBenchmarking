#!/usr/bin/env python3
import os
import sys
import glob
import argparse
import json
import pickle
from functools import reduce
from typing import List, Optional, Set

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    brier_score_loss,
)

try:
    import xgboost as xgb
    _HAVE_XGB = True
except Exception:
    _HAVE_XGB = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AlphaGenome xQTL CV from tall feature CSVs (id,feature_name,delta_*,foldchange_*) using chromosome-based CV."
    )
    p.add_argument(
        "--mode",
        type=str,
        choices=["train", "predict"],
        default="train",
        help="Run mode: train (CV + save final model) or predict (use trained model).",
    )
    p.add_argument(
        "--csv_glob",
        type=str,
        required=True,
        help="Glob of input CSV files (must contain columns: id, feature_name, and metric columns).",
    )
    p.add_argument(
        "--labels_csv",
        type=str,
        default=None,
        help="Labels CSV with columns [id,label] (or override with --id_col/--label_col).",
    )
    p.add_argument("--id_col", type=str, default="id", help="Label id column name")
    p.add_argument("--label_col", type=str, default="label", help="Binary label column name")
    p.add_argument(
        "--metrics",
        type=str,
        default="delta_mean",
        help="Comma-separated metrics to use as features. Choices subset of: delta_mean,delta_max,foldchange_mean,foldchange_max. Default: delta_mean",
    )
    p.add_argument("--folds", type=int, default=5, help="Number of chromosome-based folds used only for auto-tuning on the train split (default: 5).")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory")
    p.add_argument(
        "--model",
        type=str,
        choices=["rf", "xgb", "l1", "enet"],
        default="rf",
        help="Model type: rf, xgb, l1 (L1 logistic), enet (elastic-net logistic).",
    )
    p.add_argument("--n_jobs", type=int, default=-1, help="RandomForest parallel jobs")
    p.add_argument("--auto_tune", dest="auto_tune", action="store_true", default=True, help="Enable train-split auto tuning with chromosome-based internal folds defined by --folds (default: on).")
    p.add_argument("--no_auto_tune", dest="auto_tune", action="store_false", help="Disable train-split auto tuning and use built-in model defaults.")
    p.add_argument("--random_state", type=int, default=0, help="Base random seed")
    p.add_argument(
        "--chrom_col",
        type=str,
        default="chrom",
        help="Chromosome column name (if present). If missing, will attempt to derive from id (prefix before ':' or '_').",
    )
    p.add_argument(
        "--chunksize",
        type=int,
        default=200000,
        help="Read CSVs in chunks to filter by labels early; larger chunks are faster but use more memory (0 disables chunking).",
    )
    p.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Model directory for --mode predict: model_meta_{mt}.json, model_{mt}.json|pkl, scaler_{mt}.pkl (xgb/l1/enet). Use --model matching training.",
    )
    p.add_argument(
        "--predict_out",
        type=str,
        default=None,
        help="Prediction CSV basename under --out_dir; default: predictions_{model}.csv from saved model.",
    )
    return p.parse_args()


REQUIRED_COLS = {"id", "feature_name"}
ALL_METRICS = {"delta_mean", "delta_max", "foldchange_mean", "foldchange_max"}
DEFAULT_XGB_ROUNDS = 100
DEFAULT_XGB_EARLY_STOPPING_ROUNDS = 30
DEFAULT_XGB_GAMMA = 0.0
DEFAULT_RF_N_ESTIMATORS = 100
DEFAULT_LR_C = 0.1
DEFAULT_ENET_L1_RATIO = 0.5


def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _out_name(filename: str, model_tag: str) -> str:
    """e.g. summary.txt + rf -> summary_rf.txt"""
    base, ext = os.path.splitext(filename)
    return f"{base}_{model_tag}{ext}" if ext else f"{filename}_{model_tag}"


def _parameter_candidates(grid: dict, budget: int, random_state: int) -> list[dict]:
    total = 1
    for values in grid.values():
        total *= len(values)
    n_iter = min(total, budget)
    return list(ParameterSampler(grid, n_iter=n_iter, random_state=random_state))


def _resolve_model_params(model: str, tuned: dict, y_train: np.ndarray | None = None) -> dict:
    params = {
        "xgb_rounds": int(tuned.get("xgb_rounds", DEFAULT_XGB_ROUNDS)),
        "xgb_eta": float(tuned.get("xgb_eta", 0.1)),
        "xgb_max_depth": int(tuned.get("xgb_max_depth", 6)),
        "xgb_min_child_weight": int(tuned.get("xgb_min_child_weight", 1)),
        "xgb_subsample": float(tuned.get("xgb_subsample", 1.0)),
        "xgb_colsample_bytree": float(tuned.get("xgb_colsample_bytree", 1.0)),
        "xgb_lambda": float(tuned.get("xgb_lambda", 10.0)),
        "xgb_alpha": float(tuned.get("xgb_alpha", 0.0)),
        "xgb_gamma": float(tuned.get("xgb_gamma", DEFAULT_XGB_GAMMA)),
        "xgb_scale_pos_weight": None,
        "n_estimators": int(tuned.get("n_estimators", DEFAULT_RF_N_ESTIMATORS)),
        "max_depth": tuned.get("max_depth", None),
        "min_samples_leaf": int(tuned.get("min_samples_leaf", 1)),
        "lr_c": float(tuned.get("lr_c", DEFAULT_LR_C)),
        "enet_l1_ratio": float(tuned.get("enet_l1_ratio", DEFAULT_ENET_L1_RATIO)),
    }
    if model == "xgb":
        pos = int(np.sum(y_train == 1)) if y_train is not None else 0
        neg = int(np.sum(y_train == 0)) if y_train is not None else 0
        params["xgb_scale_pos_weight"] = float(neg / max(pos, 1)) if y_train is not None else None
    return params


def _auto_tune_params(
    args: argparse.Namespace,
    train_df: pd.DataFrame,
    chrom_col: str,
    X: np.ndarray,
    y: np.ndarray,
) -> dict:
    if not args.auto_tune:
        return {}
    tune_folds = get_chromosome_folds(train_df, chrom_col=chrom_col, n_folds=args.folds)
    if not tune_folds:
        print("WARNING: auto tuning skipped because train-split chromosome folds could not be built.", file=sys.stderr)
        return {}
    best_auc = -np.inf
    best_params = {}

    if args.model == "xgb":
        grid = {
            "xgb_rounds": [100],
            "xgb_eta": [0.03, 0.05],
            "xgb_max_depth": [3, 4],
            "xgb_min_child_weight": [1, 5],
            "xgb_subsample": [0.8],
            "xgb_colsample_bytree": [0.8],
            "xgb_lambda": [1.0, 50.0, 100.0],
            "xgb_alpha": [0.0, 0.5, 1.0],
            "xgb_gamma": [0.0, 0.5, 1.0],
        }
    elif args.model == "rf":
        grid = {"n_estimators": [100], "max_depth": [None, 10], "min_samples_leaf": [1, 5]}
    elif args.model == "l1":
        grid = {"lr_c": [1e-2, 1e-1, 1]}
    elif args.model == "enet":
        grid = {"lr_c": [1e-2, 1e-1, 1], "enet_l1_ratio": [0.5, 0.7]}
    else:
        grid = {"lr_c": [1e-2, 1e-1, 1]}
    samples = _parameter_candidates(grid, budget=20, random_state=args.random_state)
    for cand in samples:
        fold_aucs = []
        for tr_idx, te_idx in tune_folds:
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            if args.model == "xgb":
                dtr = xgb.DMatrix(X_tr, label=y_tr)
                dte = xgb.DMatrix(X_te, label=y_te)
                params = {
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "verbosity": 0,
                    "seed": args.random_state,
                    "eta": cand["xgb_eta"],
                    "max_depth": int(cand["xgb_max_depth"]),
                    "min_child_weight": int(cand["xgb_min_child_weight"]),
                    "subsample": cand["xgb_subsample"],
                    "colsample_bytree": cand["xgb_colsample_bytree"],
                    "lambda": cand["xgb_lambda"],
                    "alpha": cand["xgb_alpha"],
                    "gamma": cand["xgb_gamma"],
                }
                model = xgb.train(
                    params,
                    dtr,
                    num_boost_round=int(cand["xgb_rounds"]),
                    evals=[(dte, "valid")],
                    early_stopping_rounds=DEFAULT_XGB_EARLY_STOPPING_ROUNDS,
                    verbose_eval=False,
                )
                proba = model.predict(dte)
            elif args.model == "rf":
                clf = RandomForestClassifier(
                    n_estimators=int(cand["n_estimators"]),
                    max_depth=cand["max_depth"],
                    min_samples_leaf=int(cand["min_samples_leaf"]),
                    class_weight="balanced",
                    n_jobs=args.n_jobs,
                    random_state=args.random_state,
                )
                clf.fit(X_tr, y_tr)
                proba = clf.predict_proba(X_te)[:, 1]
            elif args.model == "l1":
                clf = LogisticRegression(
                    penalty="l1",
                    solver="saga",
                    C=float(cand["lr_c"]),
                    class_weight="balanced",
                    max_iter=100,
                    random_state=args.random_state,
                    n_jobs=args.n_jobs,
                )
                clf.fit(X_tr, y_tr)
                proba = clf.predict_proba(X_te)[:, 1]
            elif args.model == "enet":
                clf = LogisticRegression(
                    penalty="elasticnet",
                    l1_ratio=float(cand["enet_l1_ratio"]),
                    solver="saga",
                    C=float(cand["lr_c"]),
                    class_weight="balanced",
                    max_iter=100,
                    random_state=args.random_state,
                    n_jobs=args.n_jobs,
                )
                clf.fit(X_tr, y_tr)
                proba = clf.predict_proba(X_te)[:, 1]
            else:
                raise ValueError(f"Unsupported model for tuning: {args.model}")
            try:
                fold_aucs.append(float(roc_auc_score(y_te, proba)))
            except Exception:
                pass
        if fold_aucs:
            mean_auc = float(np.mean(fold_aucs))
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_params = cand
    if best_params:
        print(f"Auto-tuning on the train split with chromosome-based internal {args.folds}-fold CV evaluated {len(samples)} candidates; best AUC={best_auc:.6f}, best_params={best_params}")
    return best_params


def get_model_feature_importance(model, model_type: str, num_features: int) -> np.ndarray:
    if model_type == "xgb":
        imp_vec = np.zeros(num_features, dtype=float)
        score = model.get_score(importance_type="gain")
        for key, value in score.items():
            if key.startswith("f"):
                idx = int(key[1:])
                if 0 <= idx < num_features:
                    imp_vec[idx] = float(value)
        return imp_vec
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)
    if hasattr(model, "coef_"):
        return np.abs(np.asarray(model.coef_, dtype=float)).ravel()
    return np.zeros(num_features, dtype=float)


def read_all_feature_csvs(csv_glob: str, metrics: List[str], label_ids: Optional[Set[str]] = None, chunksize: int = 200000) -> pd.DataFrame:
    paths = sorted(glob.glob(csv_glob))
    if not paths:
        print(f"ERROR: No files matched --csv_glob {csv_glob}", file=sys.stderr)
        sys.exit(1)

    frames: List[pd.DataFrame] = []
    for pth in paths:
        try:
            # Plan read: restrict columns and optionally chunk
            # First pass to detect available metric columns quickly
            head_df = pd.read_csv(pth, nrows=1)
            missing = REQUIRED_COLS - set(head_df.columns)
            if missing:
                print(f"WARNING: {pth} missing required columns {missing}; skipping", file=sys.stderr)
                continue
            avail_metrics = [m for m in metrics if m in head_df.columns]
            if not avail_metrics:
                print(f"WARNING: {pth} has none of requested metrics {metrics}; skipping", file=sys.stderr)
                continue
            usecols = ["id", "feature_name"] + avail_metrics
            if chunksize and chunksize > 0:
                for chunk in pd.read_csv(pth, usecols=usecols, chunksize=chunksize):
                    if label_ids is not None:
                        # Ensure id as str and filter
                        chunk["id"] = chunk["id"].astype(str)
                        chunk = chunk[chunk["id"].isin(label_ids)]
                    if not chunk.empty:
                        frames.append(chunk)
            else:
                df = pd.read_csv(pth, usecols=usecols)
                if label_ids is not None:
                    df["id"] = df["id"].astype(str)
                    df = df[df["id"].isin(label_ids)]
                if not df.empty:
                    frames.append(df)
        except Exception as e:
            print(f"WARNING: failed to read {pth}: {e}", file=sys.stderr)
    if not frames:
        print("ERROR: No readable CSVs with required columns.", file=sys.stderr)
        sys.exit(1)
    # Directly concatenate without aggregation (assumes no duplicate (id, feature_name))
    tall = pd.concat(frames, axis=0, ignore_index=True)
    return tall


def pivot_to_wide(grouped: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    # Robust pivot that tolerates duplicate (id,feature_name) by aggregating mean
    present_metrics = [m for m in metrics if m in grouped.columns]
    if not present_metrics:
        print("ERROR: None of the requested metrics are present in the input.", file=sys.stderr)
        sys.exit(1)
    # Use pivot_table with aggfunc='mean' to handle duplicates safely
    wide = grouped.pivot_table(
        index="id",
        columns="feature_name",
        values=present_metrics,
        aggfunc="mean",
        observed=False,
    )
    # Flatten columns: "metric:feature"
    wide.columns = [f"{col[0]}:{col[1]}" for col in wide.columns.to_flat_index()]
    wide.reset_index(inplace=True)
    return wide


def build_dataset(
    csv_glob: str,
    labels_csv: str,
    id_col: str,
    label_col: str,
    metrics: List[str],
    chunksize: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str], List[str]]:
    # Read labels first to filter feature rows early
    labels = pd.read_csv(labels_csv, sep=None, engine="python")
    if id_col not in labels.columns or label_col not in labels.columns:
        print("ERROR: labels_csv must contain id and label columns", file=sys.stderr)
        sys.exit(1)
    # Build label id set for early filtering
    label_ids_set: Set[str] = set(labels[id_col].astype(str).values.tolist())

    grouped = read_all_feature_csvs(csv_glob, metrics, label_ids=label_ids_set, chunksize=chunksize)
    wide = pivot_to_wide(grouped, metrics)
    # Bring through potential chromosome columns from labels if present
    chrom_like = [c for c in ["chrom", "chr"] if c in labels.columns]
    label_cols = [id_col, label_col] + chrom_like
    merged = pd.merge(labels[label_cols], wide, left_on=id_col, right_on="id", how="inner")
    if merged.empty:
        print("ERROR: After merging features with labels by id, no rows remain.", file=sys.stderr)
        sys.exit(1)
    # Prepare feature matrix (exclude known meta columns)
    non_feature_cols = {id_col, label_col, "id", "chrom", "chr", "pos", "position", "ref", "alt"}
    feature_cols = [c for c in merged.columns if c not in non_feature_cols]
    X_df = merged[feature_cols].copy()
    # Fill missing with 0 for RF compatibility
    X_df = X_df.fillna(0.0)
    # Use float32 to reduce memory and speed up training
    X = X_df.astype(np.float32).values
    y = merged[label_col].astype(int).values
    ids = merged[id_col].astype(str).tolist()
    return merged, X, y, ids, feature_cols


def build_feature_matrix_for_prediction(
    csv_glob: str,
    metrics: List[str],
    chunksize: int,
) -> tuple[pd.DataFrame, np.ndarray, List[str], List[str]]:
    grouped = read_all_feature_csvs(csv_glob, metrics, label_ids=None, chunksize=chunksize)
    wide = pivot_to_wide(grouped, metrics)
    ids = wide["id"].astype(str).tolist()
    feature_cols = [c for c in wide.columns if c != "id"]
    X_df = wide[feature_cols].copy().fillna(0.0)
    X = X_df.astype(np.float32).values
    return wide, X, ids, feature_cols


def get_chromosome_folds(df: pd.DataFrame, chrom_col: str, n_folds: int) -> list:
    """
    Balanced assignment of chromosomes by sample counts, then return list of
    (train_idx, test_idx) tuples. Folds with no samples in either split are skipped.
    """
    if chrom_col not in df.columns:
        raise ValueError(f"Chromosome column '{chrom_col}' not found in input DataFrame.")
    chrom_counts = df[chrom_col].astype(str).value_counts().to_dict()
    chrom_samples = sorted(chrom_counts.items(), key=lambda x: x[1], reverse=True)
    fold_assignments = {i: [] for i in range(n_folds)}
    fold_sample_counts = [0] * n_folds
    for chrom, count in chrom_samples:
        k = fold_sample_counts.index(min(fold_sample_counts))
        fold_assignments[k].append(chrom)
        fold_sample_counts[k] += count
    folds = []
    chrom_arr = df[chrom_col].astype(str).values
    for k in range(n_folds):
        test_chroms = set(fold_assignments[k])
        if not test_chroms:
            continue
        test_mask = np.isin(chrom_arr, list(test_chroms))
        te_idx = np.where(test_mask)[0]
        tr_idx = np.where(~test_mask)[0]
        if len(te_idx) > 0 and len(tr_idx) > 0:
            folds.append((tr_idx, te_idx))
    return folds


def get_chromosome_train_test_split(df: pd.DataFrame, chrom_col: str) -> tuple[np.ndarray, np.ndarray]:
    holdout_folds = get_chromosome_folds(df, chrom_col=chrom_col, n_folds=5)
    if not holdout_folds:
        raise ValueError("Chromosome-based train/test split produced no valid folds.")
    train_idx, test_idx = holdout_folds[0]
    return train_idx, test_idx


def compute_binary_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    return {
        "n_test": int(len(y_true)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "brier": float(brier_score_loss(y_true, proba)),
    }


def main():
    args = parse_args()
    ensure_out_dir(args.out_dir)
    if args.model == "xgb" and not _HAVE_XGB:
        print("ERROR: --model xgb requested but xgboost is not available.", file=sys.stderr)
        sys.exit(1)

    # Metrics list validation
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    for m in metrics:
        if m not in ALL_METRICS:
            print(f"ERROR: metric '{m}' not in allowed set {sorted(list(ALL_METRICS))}", file=sys.stderr)
            sys.exit(1)

    if args.mode == "train" and not args.labels_csv:
        print("ERROR: --mode train requires --labels_csv.", file=sys.stderr)
        sys.exit(1)

    if args.mode == "predict":
        if not args.model_dir:
            print("ERROR: --mode predict requires --model_dir.", file=sys.stderr)
            sys.exit(1)
        meta_path = os.path.join(args.model_dir, _out_name("model_meta.json", args.model))
        if not os.path.isfile(meta_path):
            print(
                f"ERROR: missing {meta_path} (use --model matching training; expected model_meta_{{rf|xgb|l1|enet}}.json).",
                file=sys.stderr,
            )
            sys.exit(1)

        with open(meta_path, "r") as f:
            meta = json.load(f)
        model_type = meta["model_type"]
        if model_type != args.model:
            print(
                f"ERROR: --model {args.model} but metadata has model_type={model_type}.",
                file=sys.stderr,
            )
            sys.exit(1)
        train_metrics = meta["metrics"]
        trained_feature_cols = meta["feature_cols"]

        _, X_new, ids_new, feature_cols_new = build_feature_matrix_for_prediction(
            args.csv_glob, train_metrics, args.chunksize
        )

        # Align incoming features to training feature order:
        # missing columns -> 0, extra columns -> dropped.
        new_col_to_idx = {c: i for i, c in enumerate(feature_cols_new)}
        X_aligned = np.zeros((X_new.shape[0], len(trained_feature_cols)), dtype=np.float32)
        for j, c in enumerate(trained_feature_cols):
            idx = new_col_to_idx.get(c)
            if idx is not None:
                X_aligned[:, j] = X_new[:, idx]

        if model_type == "xgb":
            scaler_path = os.path.join(args.model_dir, _out_name("scaler.pkl", model_type))
            if not os.path.isfile(scaler_path):
                print(f"ERROR: missing scaler file {scaler_path}", file=sys.stderr)
                sys.exit(1)
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            X_aligned = scaler.transform(X_aligned)
            model_path = os.path.join(args.model_dir, _out_name("model.json", model_type))
            if not os.path.isfile(model_path):
                print(f"ERROR: missing xgb model file {model_path}", file=sys.stderr)
                sys.exit(1)
            model = xgb.Booster()
            model.load_model(model_path)
            dnew = xgb.DMatrix(X_aligned)
            proba = model.predict(dnew)
        else:
            if model_type in {"l1", "enet"}:
                scaler_path = os.path.join(args.model_dir, _out_name("scaler.pkl", model_type))
                if not os.path.isfile(scaler_path):
                    print(f"ERROR: missing scaler file {scaler_path}", file=sys.stderr)
                    sys.exit(1)
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
                X_aligned = scaler.transform(X_aligned)
            model_path = os.path.join(args.model_dir, _out_name("model.pkl", model_type))
            if not os.path.isfile(model_path):
                print(f"ERROR: missing model file {model_path}", file=sys.stderr)
                sys.exit(1)
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            proba = model.predict_proba(X_aligned)[:, 1]

        thr = float(meta.get("threshold", 0.5))
        pred = (proba >= thr).astype(int)
        pred_df = pd.DataFrame({"id": ids_new, "y_prob": proba, "y_pred": pred})
        pred_bn = args.predict_out if args.predict_out else _out_name("predictions.csv", model_type)
        pred_path = os.path.join(args.out_dir, pred_bn)
        pred_df.to_csv(pred_path, index=False)
        print(f"Prediction completed. Results written to: {pred_path}")
        return

    merged, X, y, ids, feature_cols = build_dataset(
        args.csv_glob, args.labels_csv, args.id_col, args.label_col, metrics, args.chunksize
    )
    mt = args.model  # output filename tag: rf or xgb

    # Determine chromosome column for chromosome-based CV; derive from id if needed
    chrom_col = None
    if args.chrom_col in merged.columns:
        chrom_col = args.chrom_col
    elif "chrom" in merged.columns:
        chrom_col = "chrom"
    elif "chr" in merged.columns:
        chrom_col = "chr"
    else:
        # Derive from id by splitting 'chr_pos_ref_alt' with underscore
        id_parts = merged["id"].astype(str).str.split("_", expand=True)
        if isinstance(id_parts, pd.DataFrame) and id_parts.shape[1] >= 4:
            merged["chrom"] = id_parts.iloc[:, 0]
            merged["pos"] = id_parts.iloc[:, 1]
            merged["ref"] = id_parts.iloc[:, 2]
            merged["alt"] = id_parts.iloc[:, 3]
        else:
            # Fallback: use the first token as chromosome
            merged["chrom"] = merged["id"].astype(str).str.split("_").str[0]
        chrom_col = "chrom"
    try:
        train_idx, test_idx = get_chromosome_train_test_split(merged, chrom_col=chrom_col)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    train_df = merged.iloc[train_idx].reset_index(drop=True)
    test_df = merged.iloc[test_idx].reset_index(drop=True)
    X_train = X[train_idx]
    y_train = y[train_idx]
    ids_train = [ids[i] for i in train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    ids_test = [ids[i] for i in test_idx]

    scaler = None
    if args.model in {"xgb", "l1", "enet"}:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        np.savetxt(os.path.join(args.out_dir, _out_name("scaler_mean.txt", mt)), scaler.mean_, fmt="%.6f")
        np.savetxt(os.path.join(args.out_dir, _out_name("scaler_scale.txt", mt)), scaler.scale_, fmt="%.6f")

    tuned = _auto_tune_params(args, train_df, chrom_col, X_train, y_train)
    model_params = _resolve_model_params(args.model, tuned, y_train)

    num_features = X_train.shape[1]
    # Train the final model on the full training split and use it for both
    # held-out test evaluation and feature importance export.
    if args.model == "xgb":
        dtrain_all = xgb.DMatrix(X_train, label=y_train)
        params = {
            "objective": "binary:logistic",
            "eta": model_params["xgb_eta"],
            "max_depth": model_params["xgb_max_depth"],
            "min_child_weight": model_params["xgb_min_child_weight"],
            "subsample": model_params["xgb_subsample"],
            "colsample_bytree": model_params["xgb_colsample_bytree"],
            "lambda": model_params["xgb_lambda"],
            "alpha": model_params["xgb_alpha"],
            "gamma": model_params["xgb_gamma"],
            "scale_pos_weight": model_params["xgb_scale_pos_weight"],
            "eval_metric": "logloss",
            "verbosity": 0,
            "seed": args.random_state,
        }
        final_model = xgb.train(params, dtrain_all, num_boost_round=model_params["xgb_rounds"])
        final_model.save_model(os.path.join(args.out_dir, _out_name("model.json", mt)))
        if scaler is not None:
            with open(os.path.join(args.out_dir, _out_name("scaler.pkl", mt)), "wb") as f:
                pickle.dump(scaler, f)
        test_proba = final_model.predict(xgb.DMatrix(X_test))
    else:
        if args.model == "rf":
            final_model = RandomForestClassifier(
                n_estimators=model_params["n_estimators"],
                max_depth=model_params["max_depth"],
                min_samples_leaf=model_params["min_samples_leaf"],
                class_weight="balanced",
                n_jobs=args.n_jobs,
                random_state=args.random_state,
            )
        elif args.model == "l1":
            final_model = LogisticRegression(
                penalty="l1",
                solver="saga",
                C=model_params["lr_c"],
                class_weight="balanced",
                max_iter=100,
                random_state=args.random_state,
                n_jobs=args.n_jobs,
            )
        elif args.model == "enet":
            final_model = LogisticRegression(
                penalty="elasticnet",
                l1_ratio=model_params["enet_l1_ratio"],
                solver="saga",
                C=model_params["lr_c"],
                class_weight="balanced",
                max_iter=100,
                random_state=args.random_state,
                n_jobs=args.n_jobs,
            )
        else:
            raise ValueError(f"Unsupported model: {args.model}")
        final_model.fit(X_train, y_train)
        with open(os.path.join(args.out_dir, _out_name("model.pkl", mt)), "wb") as f:
            pickle.dump(final_model, f)
        if scaler is not None and args.model in {"l1", "enet"}:
            with open(os.path.join(args.out_dir, _out_name("scaler.pkl", mt)), "wb") as f:
                pickle.dump(scaler, f)
        test_proba = final_model.predict_proba(X_test)[:, 1]

    final_importance = get_model_feature_importance(final_model, args.model, num_features)
    pd.DataFrame({"feature": feature_cols, "importance": final_importance}).to_csv(
        os.path.join(args.out_dir, "feature_importances.csv"), index=False
    )

    test_pred = (test_proba >= 0.5).astype(int)
    test_summary = compute_binary_metrics(y_test, test_proba)
    test_summary["train_size"] = int(len(train_idx))
    test_summary["test_size"] = int(len(test_idx))
    test_summary["train_chromosomes"] = int(train_df[chrom_col].astype(str).nunique())
    test_summary["test_chromosomes"] = int(test_df[chrom_col].astype(str).nunique())
    with open(os.path.join(args.out_dir, "test_summary.txt"), "w") as f:
        for k, v in test_summary.items():
            f.write(f"{k}\t{v}\n")

    test_pred_df = pd.DataFrame(
        {"id": ids_test, "y_true": y_test.astype(int), "y_prob": test_proba, "y_pred": test_pred}
    )
    test_pred_df.to_csv(os.path.join(args.out_dir, "test_predictions.csv"), index=False)

    model_meta = {
        "model_type": args.model,
        "metrics": metrics,
        "feature_cols": feature_cols,
        "id_col": args.id_col,
        "label_col": args.label_col,
        "threshold": 0.5,
        "lr_c": model_params["lr_c"],
        "enet_l1_ratio": model_params["enet_l1_ratio"],
        "n_estimators": model_params["n_estimators"],
        "max_depth": model_params["max_depth"],
        "min_samples_leaf": model_params["min_samples_leaf"],
        "xgb_rounds": model_params["xgb_rounds"],
        "xgb_eta": model_params["xgb_eta"],
        "xgb_max_depth": model_params["xgb_max_depth"],
        "xgb_min_child_weight": model_params["xgb_min_child_weight"],
        "xgb_subsample": model_params["xgb_subsample"],
        "xgb_colsample_bytree": model_params["xgb_colsample_bytree"],
        "xgb_lambda": model_params["xgb_lambda"],
        "xgb_alpha": model_params["xgb_alpha"],
        "xgb_gamma": model_params["xgb_gamma"],
        "xgb_early_stopping_rounds": DEFAULT_XGB_EARLY_STOPPING_ROUNDS,
        "xgb_scale_pos_weight": model_params["xgb_scale_pos_weight"],
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "tuning_folds_on_train": int(args.folds),
        "test_split_strategy": "chromosome_holdout_20_percent",
        "auto_tune": args.auto_tune,
        "auto_tune_strategy": f"train_split_chromosome_internal_{args.folds}_fold",
        "auto_tune_budget": 20,
        "best_params": tuned,
    }
    with open(os.path.join(args.out_dir, _out_name("model_meta.json", mt)), "w") as f:
        json.dump(model_meta, f, indent=2)

    print(f"Completed AlphaGenome xQTL training with chromosome holdout test evaluation. Outputs at: {args.out_dir}")
    print(f"Final model and metadata saved under: {args.out_dir}")


if __name__ == "__main__":
    main()

