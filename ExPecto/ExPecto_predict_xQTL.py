#!/usr/bin/env python3
import os
import sys
import glob
import re
import argparse
import json
import pickle
import numpy as np
import pandas as pd
import h5py
from typing import List, Tuple, Optional
from functools import reduce

from sklearn.model_selection import ParameterSampler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

try:
    import xgboost as xgb
    _HAVE_XGB = True
except Exception:
    _HAVE_XGB = False
from sklearn.ensemble import RandomForestClassifier


MODEL_DEFAULTS = {
    "xgb_rounds": 100,
    "xgb_early_stopping_rounds": 30,
    "xgb_eta": 0.1,
    "xgb_max_depth": 6,
    "xgb_min_child_weight": 1,
    "xgb_subsample": 1.0,
    "xgb_colsample_bytree": 1.0,
    "xgb_lambda": 10.0,
    "xgb_alpha": 0.0,
    "xgb_gamma": 0.0,
    "xgb_scale_pos_weight": "auto",
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_leaf": 1,
    "lr_c": 1.0,
    "enet_l1_ratio": 0.5,
}


def _initialize_model_defaults(args: argparse.Namespace, y_train: Optional[np.ndarray] = None) -> None:
    for key, value in MODEL_DEFAULTS.items():
        setattr(args, key, value)
    if y_train is not None:
        pos = int(np.sum(y_train == 1))
        neg = int(np.sum(y_train == 0))
        args.xgb_scale_pos_weight = float(neg / max(pos, 1))




def _sample_tuning_candidates(grid: dict, max_trials: int, seed: int) -> List[dict]:
    total = 1
    for values in grid.values():
        total *= len(values)
    if total <= max_trials:
        return list(ParameterSampler(grid, n_iter=total, random_state=seed))
    return list(ParameterSampler(grid, n_iter=max_trials, random_state=seed))


def _auto_tune_selected_model(
    args,
    X: np.ndarray,
    y: np.ndarray,
    chroms: pd.Series,
) -> dict:
    if not getattr(args, "auto_tune", True):
        return {}
    counts = np.bincount(y.astype(int))
    if counts.size < 2 or np.min(counts) < 2:
        print("WARNING: auto tuning skipped due to insufficient class samples.", file=sys.stderr)
        return {}
    cv = get_chromosome_folds(chroms, n_folds=args.folds, seed=args.random_state)
    if not cv:
        print(
            f"WARNING: auto tuning skipped because train split chromosome-based {args.folds}-fold construction produced no valid folds.",
            file=sys.stderr,
        )
        return {}
    best_auc = -1.0
    best = {}
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
    candidates = _sample_tuning_candidates(grid, max_trials=20, seed=args.random_state)
    tune_n_jobs = int(getattr(args, "n_jobs", -1))
    print(
        f"Auto-tuning {args.model} on train split using chromosome-based {args.folds}-fold CV ({len(candidates)} parameter sets)."
    )
    for cand in candidates:
        aucs = []
        for tr_idx, te_idx in cv:
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            if args.model == "xgb":
                dtr = xgb.DMatrix(X_tr, label=y_tr)
                dte = xgb.DMatrix(X_te, label=y_te)
                p = {
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
                    "scale_pos_weight": args.xgb_scale_pos_weight,
                }
                m = xgb.train(p, dtr, num_boost_round=int(cand["xgb_rounds"]), evals=[(dte, "valid")], early_stopping_rounds=args.xgb_early_stopping_rounds, verbose_eval=False)
                proba = m.predict(dte)
            elif args.model == "rf":
                m = RandomForestClassifier(n_estimators=int(cand["n_estimators"]), max_depth=cand["max_depth"], min_samples_leaf=int(cand["min_samples_leaf"]), max_features="log2", class_weight="balanced", n_jobs=tune_n_jobs, random_state=args.random_state)
                m.fit(X_tr, y_tr)
                proba = m.predict_proba(X_te)[:, 1]
            elif args.model == "l1":
                m = LogisticRegression(penalty="l1", solver="saga", C=float(cand["lr_c"]), class_weight="balanced", max_iter=100, random_state=args.random_state, n_jobs=tune_n_jobs)
                m.fit(X_tr, y_tr)
                proba = m.predict_proba(X_te)[:, 1]
            elif args.model == "enet":
                m = LogisticRegression(penalty="elasticnet", l1_ratio=float(cand["enet_l1_ratio"]), solver="saga", C=float(cand["lr_c"]), class_weight="balanced", max_iter=100, random_state=args.random_state, n_jobs=tune_n_jobs)
                m.fit(X_tr, y_tr)
                proba = m.predict_proba(X_te)[:, 1]
            try:
                aucs.append(float(roc_auc_score(y_te, proba)))
            except Exception:
                pass
        if aucs:
            auc = float(np.mean(aucs))
            if auc > best_auc:
                best_auc = auc
                best = cand
    for k, v in best.items():
        setattr(args, k, v)
    if best:
        print(f"Auto-tuning best AUC={best_auc:.6f}, best_params={best}")
    return best


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="xQTL prediction using ExPecto chromatin diffs (.diff.h5) with chromosome-based train/test split and train-only chromosome tuning CV"
    )
    p.add_argument(
        "--mode",
        type=str,
        choices=["train", "predict"],
        default="train",
        help="train: CV + save final model; predict: score new VCF + matching .diff.h5",
    )
    p.add_argument("--vcf", type=str, required=True, help="Input VCF used by chromatin.py (same order)")
    p.add_argument(
        "--h5_glob",
        type=str,
        required=True,
        help="Glob pattern for chromatin.py outputs, e.g. '/path/to/input.vcf.shift_*.diff.h5'",
    )
    p.add_argument(
        "--labels_csv",
        type=str,
        default=None,
        help="CSV with id,label columns (required for --mode train)",
    )
    p.add_argument("--id_col", type=str, default="id", help="Label id column name")
    p.add_argument("--label_col", type=str, default="label", help="Binary label column name")
    p.add_argument(
        "--feature_meta",
        type=str,
        required=False,
        help="Optional feature metadata file (TSV/CSV) with columns: number, Cell type, Assay, Treatment, Assay type, Source, link. "
             "Feature names will be built as 'Cell|type_Assay|Treatment' in the order given by 'number'.",
    )
    p.add_argument("--dist_strand_csv", type=str, required=False, default=None,
                   help="Optional CSV with columns: id, dist, strand (+/-). If provided, build ExPecto positional kernels; otherwise fall back to --agg over shifts.")
    p.add_argument(
        "--agg",
        type=str,
        choices=["mean", "max", "concat"],
        default="mean",
        help="Aggregate across shifts: mean/max over shifts or concatenate",
    )
    p.add_argument("--folds", type=int, default=5, help="Number of chromosome-based folds used for auto-tuning on the training split.")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory")
    p.add_argument(
        "--model",
        type=str,
        choices=["xgb", "rf", "l1", "enet"],
        default="xgb",
        help="Classifier family: xgb, rf, l1 (L1 logistic), enet (elastic-net logistic). Model-specific hyperparameters come from internal defaults plus auto-tune outputs.",
    )
    p.add_argument("--random_state", type=int, default=0)
    p.add_argument("--auto_tune", dest="auto_tune", action="store_true", default=True, help="Enable automatic hyperparameter tuning on the train split using internal chromosome-based folds from --folds.")
    p.add_argument("--no_auto_tune", dest="auto_tune", action="store_false", help="Disable automatic tuning and use internal model defaults for training.")
    p.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Directory with model_meta_{mt}.json, model_{mt}.pkl|json, scaler_{mt}.pkl (xgb/l1/enet). --model must match training.",
    )
    p.add_argument(
        "--predict_out",
        type=str,
        default=None,
        help="Prediction CSV basename under --out_dir; default: predictions_{model}.csv.",
    )
    return p.parse_args()


def read_vcf_core(vcf_path: str) -> pd.DataFrame:
    """Read minimal VCF columns needed: CHROM(0), POS(1), ID(2), REF(3), ALT(4)."""
    vcf = pd.read_csv(vcf_path, sep="\t", header=None, comment="#", dtype={0: str})
    vcf = vcf[[0, 1, 2, 3, 4]].copy()
    vcf.columns = ["chrom", "pos", "id", "ref", "alt"]
    return vcf


def load_h5_diffs(paths: List[str], num_variants: int) -> List[np.ndarray]:
    """Load .diff.h5 files; each may have 2*N rows (forward + reverse). Reduce to N by averaging pairs."""
    results = []
    for pth in sorted(paths):
        with h5py.File(pth, "r") as hf:
            diff = hf["pred"][...]
        if diff.shape[0] == num_variants:
            reduced = diff
        elif diff.shape[0] == 2 * num_variants:
            # average pairs: [0..N-1] with [N..2N-1]
            reduced = 0.5 * (diff[:num_variants, :] + diff[num_variants:, :])
        else:
            raise ValueError(f"Unexpected diff rows {diff.shape[0]} for N={num_variants} in {pth}")
        results.append(reduced)
    return results


def aggregate_shifts(shift_mats: List[np.ndarray], mode: str) -> np.ndarray:
    """Aggregate list of (N, F) by mean/max or concatenate into (N, F*S)."""
    if len(shift_mats) == 1:
        return shift_mats[0]
    if mode == "mean":
        return np.mean(shift_mats, axis=0)
    if mode == "max":
        return np.max(shift_mats, axis=0)
    if mode == "concat":
        return np.concatenate(shift_mats, axis=1)
    raise ValueError("Unknown agg mode")


def parse_shifts_from_paths(paths: List[str]) -> List[int]:
    """Extract integer shift values from filenames containing 'shift_###'."""
    shifts = []
    for p in sorted(paths):
        m = re.search(r"shift_(-?\d+)", os.path.basename(p))
        if not m:
            raise ValueError("Cannot parse shift value from filename: %s" % p)
        shifts.append(int(m.group(1)))
    return shifts


def build_expecto_position_kernels(dist: np.ndarray, strand: np.ndarray, shifts: List[int]) -> List[np.ndarray]:
    """
    Build ExPecto-like positional kernels per shift.
    Returns list of arrays with shape (N, 10) for each shift:
    5 decays (scales=[0.01,0.02,0.05,0.1,0.2]) for downstream (<=0) and 5 for upstream (>=0).
    """
    sign = np.where(strand.astype(str) == '+', 1, -1).astype(np.int32)
    dist = dist.astype(np.int64)
    scales = np.array([0.01, 0.02, 0.05, 0.1, 0.2], dtype=np.float64)
    kernels = []
    for d in shifts:
        eff = dist + d * sign  # N
        bins = np.floor(np.abs(eff / 200.0))
        down_mask = (eff <= 0).astype(np.float64)
        up_mask = (eff >= 0).astype(np.float64)
        down = np.stack([np.exp(-s * bins) * down_mask for s in scales], axis=1)
        up = np.stack([np.exp(-s * bins) * up_mask for s in scales], axis=1)
        W = np.concatenate([down, up], axis=1).astype(np.float32)  # N x 10
        kernels.append(W)
    return kernels


def get_chromosome_folds(chroms: pd.Series, n_folds: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    # Balanced assignment of chromosomes by sample counts
    rng = np.random.RandomState(seed)
    chrom_counts = chroms.value_counts().to_dict()
    items = list(chrom_counts.items())
    rng.shuffle(items)  # stable randomness before sorting to break ties differently per seed
    items.sort(key=lambda x: x[1], reverse=True)

    fold_assign = {i: [] for i in range(n_folds)}
    fold_sizes = [0] * n_folds
    for ch, cnt in items:
        k = fold_sizes.index(min(fold_sizes))
        fold_assign[k].append(ch)
        fold_sizes[k] += cnt

    folds = []
    chrom_arr = np.asarray(chroms.values)
    for k in range(n_folds):
        test_chs = set(fold_assign[k])
        test_idx = np.where(np.isin(chrom_arr, list(test_chs)))[0]
        train_idx = np.where(~np.isin(chrom_arr, list(test_chs)))[0]
        if len(test_idx) > 0 and len(train_idx) > 0:
            folds.append((train_idx, test_idx))
    return folds


def get_chromosome_train_test_split(chroms: pd.Series, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    holdout_folds = get_chromosome_folds(chroms, n_folds=5, seed=seed)
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
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "brier": float(np.mean((y_true - proba) ** 2)),
    }


def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _out_name(filename: str, model_tag: str) -> str:
    base, ext = os.path.splitext(filename)
    return f"{base}_{model_tag}{ext}" if ext else f"{filename}_{model_tag}"


def assign_feature_names(
    num_features: int,
    args: argparse.Namespace,
    h5_paths: List[str],
    shift_mats: List[np.ndarray],
) -> List[str]:
    """Match training logic for readable feature names from optional feature_meta."""
    feature_names = [f"f{i}" for i in range(num_features)]
    if not args.feature_meta:
        return feature_names
    try:
        meta = pd.read_csv(args.feature_meta, sep=None, engine="python")
        required_cols = {"number", "Cell type", "Assay"}
        if required_cols.issubset(set(meta.columns)):
            meta = meta.copy()
            cell_type_str = meta["Cell type"].astype(str).str.replace(" ", "|")
            assay_str = meta["Assay"].astype(str)
            if "Treatment" in meta.columns:
                treatment_str = meta["Treatment"].astype(str)
                treatment_str = treatment_str.replace({"": "None", "nan": "None", "None": "None"})
                treatment_str = treatment_str.fillna("None")
            else:
                treatment_str = pd.Series(["None"] * len(meta))
            meta = meta.assign(name=cell_type_str + "|" + assay_str + "|" + treatment_str)
            numbers = meta["number"].astype(int).values
            if args.agg == "concat":
                base_len = shift_mats[0].shape[1]
            else:
                base_len = num_features
            zero_based = (numbers.min() == 0) and (numbers.max() >= base_len - 1)
            one_based = (numbers.min() == 1) and (numbers.max() >= base_len)
            idxs = numbers.copy()
            if one_based and not zero_based:
                idxs = numbers - 1
            base_names = [f"f{i}" for i in range(base_len)]
            for k, nm in zip(idxs, meta["name"].astype(str).values):
                if 0 <= k < base_len:
                    base_names[k] = nm
            if args.agg == "concat" and num_features % base_len == 0:
                repeats = num_features // base_len
                if repeats == len(parse_shifts_from_paths(h5_paths)):
                    expanded = []
                    for _ in range(repeats):
                        expanded.extend(base_names)
                    feature_names = expanded
                else:
                    feature_names = (base_names * repeats)[:num_features]
            else:
                feature_names = base_names[:num_features]
        else:
            print("WARNING: feature_meta missing required columns; using default feature names.", file=sys.stderr)
    except Exception as e:
        print(f"WARNING: failed to parse feature_meta '{args.feature_meta}': {e}. Using default feature names.", file=sys.stderr)
    return feature_names


def build_expecto_feature_matrix(
    args: argparse.Namespace,
    vcf: pd.DataFrame,
    labels: Optional[pd.DataFrame],
    train_mode: bool,
) -> Tuple[np.ndarray, np.ndarray, pd.Series, List[str], int, pd.DataFrame]:
    """
    Load .diff.h5, merge labels (train), build X and parallel chrom_sel for CV.
    Returns: X_sel, y (empty array if predict), chrom_sel, feature_names, num_features, merged
    """
    num_variants = vcf.shape[0]
    ids = vcf["id"].astype(str)
    chroms = vcf["chrom"].astype(str)

    h5_paths = glob.glob(args.h5_glob)
    if not h5_paths:
        print(f"ERROR: No .diff.h5 matched by pattern: {args.h5_glob}", file=sys.stderr)
        sys.exit(1)
    shift_mats = load_h5_diffs(h5_paths, num_variants)
    shifts = parse_shifts_from_paths(h5_paths)

    df = pd.DataFrame({"id": ids, "chrom": chroms})
    df["row_index"] = np.arange(num_variants)

    if train_mode:
        if labels is None or args.id_col not in labels.columns or args.label_col not in labels.columns:
            print("ERROR: labels_csv must contain id and label columns", file=sys.stderr)
            sys.exit(1)
        merged = pd.merge(labels, df, left_on=args.id_col, right_on="id", how="inner")
    else:
        merged = df

    if args.dist_strand_csv:
        ds = pd.read_csv(args.dist_strand_csv)
        for col in ("id", "dist", "strand"):
            if col not in ds.columns:
                print("ERROR: dist_strand_csv must have columns: id, dist, strand", file=sys.stderr)
                sys.exit(1)
        merged = pd.merge(merged, ds[["id", "dist", "strand"]], on="id", how="inner")

    if merged.empty:
        print("ERROR: After merging inputs with VCF ids, no rows remain.", file=sys.stderr)
        sys.exit(1)

    idx = np.asarray(merged["row_index"].values)
    if train_mode:
        y = np.asarray(merged[args.label_col].values).astype(int)
    else:
        y = np.array([], dtype=int)
    chrom_sel = merged["chrom"].astype(str).reset_index(drop=True)

    if args.dist_strand_csv:
        dist = np.asarray(merged["dist"].values).astype(np.int64)
        strand = np.asarray(merged["strand"].values).astype(str)
        kernels = build_expecto_position_kernels(dist, strand, shifts)
        shift_mats_sel = [M[idx] for M in shift_mats]
        parts = []
        for Ej, Wj in zip(shift_mats_sel, kernels):
            n, f = Ej.shape
            parts.append(np.tile(Ej, 10) * np.repeat(Wj, f, axis=1))
        X_sel = reduce(lambda a, b: a + b, parts)
    else:
        X = aggregate_shifts(shift_mats, args.agg)
        X_sel = X[idx]

    num_features = X_sel.shape[1]
    feature_names = assign_feature_names(num_features, args, h5_paths, shift_mats)
    return X_sel, y, chrom_sel, feature_names, num_features, merged


def run_predict(args: argparse.Namespace) -> None:
    """Load saved model; requires new --vcf, --h5_glob, and same feature recipe as training (see model_meta)."""
    if not args.model_dir:
        print("ERROR: --mode predict requires --model_dir.", file=sys.stderr)
        sys.exit(1)
    meta_path = os.path.join(args.model_dir, _out_name("model_meta.json", args.model))
    if not os.path.isfile(meta_path):
        print(
            f"ERROR: missing {meta_path} (use --model matching training).",
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
    if model_type == "xgb" and not _HAVE_XGB:
        print("ERROR: saved model is xgb but xgboost is not available.", file=sys.stderr)
        sys.exit(1)

    args.agg = meta["agg"]
    use_ds = bool(meta.get("use_dist_strand", False))
    if use_ds and not args.dist_strand_csv:
        print("ERROR: this model was trained with --dist_strand_csv; provide the same for predict.", file=sys.stderr)
        sys.exit(1)
    if not use_ds and args.dist_strand_csv:
        print(
            "WARNING: model was trained without dist/strand; ignoring --dist_strand_csv for predict.",
            file=sys.stderr,
        )
        args.dist_strand_csv = None

    fm = meta.get("feature_meta")
    args.feature_meta = fm if fm else None

    vcf = read_vcf_core(args.vcf)
    X_sel, _y, _chrom, _fn, num_features, merged = build_expecto_feature_matrix(
        args, vcf, labels=None, train_mode=False
    )
    if int(meta.get("num_features", num_features)) != num_features:
        print(
            f"ERROR: feature dimension mismatch: model expects {meta.get('num_features')} features, got {num_features}.",
            file=sys.stderr,
        )
        sys.exit(1)

    X_in = X_sel.astype(np.float64)
    if model_type == "xgb":
        scaler_path = os.path.join(args.model_dir, _out_name("scaler.pkl", model_type))
        if not os.path.isfile(scaler_path):
            print(f"ERROR: missing {scaler_path}", file=sys.stderr)
            sys.exit(1)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        X_in = scaler.transform(X_in)
        model_path = os.path.join(args.model_dir, _out_name("model.json", model_type))
        if not os.path.isfile(model_path):
            print(f"ERROR: missing {model_path}", file=sys.stderr)
            sys.exit(1)
        booster = xgb.Booster()
        booster.load_model(model_path)
        proba = booster.predict(xgb.DMatrix(X_in))
    else:
        if model_type in {"l1", "enet"}:
            scaler_path = os.path.join(args.model_dir, _out_name("scaler.pkl", model_type))
            if not os.path.isfile(scaler_path):
                print(f"ERROR: missing {scaler_path}", file=sys.stderr)
                sys.exit(1)
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            X_in = scaler.transform(X_in)
        model_path = os.path.join(args.model_dir, _out_name("model.pkl", model_type))
        if not os.path.isfile(model_path):
            print(f"ERROR: missing {model_path}", file=sys.stderr)
            sys.exit(1)
        with open(model_path, "rb") as f:
            clf = pickle.load(f)
        proba = clf.predict_proba(X_in)[:, 1]

    thr = float(meta.get("threshold", 0.5))
    pred = (proba >= thr).astype(int)
    out_ids = merged["id"].astype(str).values
    pred_df = pd.DataFrame({"id": out_ids, "y_prob": proba, "y_pred": pred})
    pred_bn = args.predict_out if args.predict_out else _out_name("predictions.csv", model_type)
    pred_path = os.path.join(args.out_dir, pred_bn)
    pred_df.to_csv(pred_path, index=False)
    print(f"Prediction completed. Results written to: {pred_path}")


def main() -> None:
    args = parse_args()
    ensure_out_dir(args.out_dir)
    if args.mode == "train" and args.model == "xgb" and not _HAVE_XGB:
        print("ERROR: --model xgb was selected but XGBoost is not available in this environment.", file=sys.stderr)
        sys.exit(1)
    if args.mode == "train" and not args.labels_csv:
        print("ERROR: --mode train requires --labels_csv.", file=sys.stderr)
        sys.exit(1)
    if args.mode == "predict":
        run_predict(args)
        return

    vcf = read_vcf_core(args.vcf)
    labels = pd.read_csv(args.labels_csv)
    X_sel, y, chrom_sel, feature_names, num_features, merged = build_expecto_feature_matrix(
        args, vcf, labels, train_mode=True
    )
    mt = args.model

    try:
        train_idx, test_idx = get_chromosome_train_test_split(chrom_sel, seed=args.random_state)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    train_df = merged.iloc[train_idx].reset_index(drop=True)
    test_df = merged.iloc[test_idx].reset_index(drop=True)
    X_train = X_sel[train_idx]
    y_train = y[train_idx]
    X_test = X_sel[test_idx]
    y_test = y[test_idx]
    ids_test = merged["id"].iloc[test_idx].astype(str).tolist()
    chrom_train = chrom_sel.iloc[train_idx].reset_index(drop=True)

    scaler = None
    if args.model in {"xgb", "l1", "enet"}:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        np.savetxt(os.path.join(args.out_dir, _out_name("scaler_mean.txt", mt)), scaler.mean_, fmt="%.6f")
        np.savetxt(os.path.join(args.out_dir, _out_name("scaler_scale.txt", mt)), scaler.scale_, fmt="%.6f")

    _initialize_model_defaults(args, y_train=y_train)
    tuned = _auto_tune_selected_model(args, X_train, y_train, chrom_train)
    if args.model == "rf":
        setattr(args, "max_depth", tuned.get("max_depth", None))
        setattr(args, "min_samples_leaf", int(tuned.get("min_samples_leaf", 1)))
    else:
        args.lr_c = float(tuned.get("lr_c", args.lr_c))
        args.enet_l1_ratio = float(tuned.get("enet_l1_ratio", args.enet_l1_ratio))
    args.xgb_rounds = int(tuned.get("xgb_rounds", args.xgb_rounds))
    args.xgb_eta = float(tuned.get("xgb_eta", args.xgb_eta))
    args.xgb_max_depth = int(tuned.get("xgb_max_depth", args.xgb_max_depth))
    args.xgb_min_child_weight = int(tuned.get("xgb_min_child_weight", args.xgb_min_child_weight))
    args.xgb_subsample = float(tuned.get("xgb_subsample", args.xgb_subsample))
    args.xgb_colsample_bytree = float(tuned.get("xgb_colsample_bytree", args.xgb_colsample_bytree))
    args.xgb_lambda = float(tuned.get("xgb_lambda", args.xgb_lambda))
    args.xgb_alpha = float(tuned.get("xgb_alpha", args.xgb_alpha))
    args.xgb_gamma = float(tuned.get("xgb_gamma", args.xgb_gamma))

    if args.model == "xgb":
        params = {
            "objective": "binary:logistic",
            "eta": args.xgb_eta,
            "max_depth": args.xgb_max_depth,
            "min_child_weight": args.xgb_min_child_weight,
            "subsample": args.xgb_subsample,
            "colsample_bytree": args.xgb_colsample_bytree,
            "lambda": args.xgb_lambda,
            "alpha": args.xgb_alpha,
            "gamma": args.xgb_gamma,
            "scale_pos_weight": args.xgb_scale_pos_weight,
            "eval_metric": "logloss",
            "verbosity": 0,
            "seed": args.random_state,
        }
        dtrain_all = xgb.DMatrix(X_train, label=y_train)
        final_model = xgb.train(params, dtrain_all, num_boost_round=args.xgb_rounds)
        final_model.save_model(os.path.join(args.out_dir, _out_name("model.json", mt)))
        if scaler is not None:
            with open(os.path.join(args.out_dir, _out_name("scaler.pkl", mt)), "wb") as f:
                pickle.dump(scaler, f)
        final_importance = np.zeros(num_features, dtype=float)
        try:
            score_dict = final_model.get_score(importance_type="gain")
            for k, v in score_dict.items():
                if k.startswith("f"):
                    idx = int(k[1:])
                    if 0 <= idx < num_features:
                        final_importance[idx] = float(v)
        except Exception:
            pass
        test_proba = final_model.predict(xgb.DMatrix(X_test))
    elif args.model == "rf":
        final_model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=getattr(args, "max_depth", None),
            min_samples_leaf=int(getattr(args, "min_samples_leaf", 1)),
            max_features="log2",
            class_weight="balanced",
            n_jobs=-1,
            random_state=args.random_state,
        )
        final_model.fit(X_train, y_train)
        with open(os.path.join(args.out_dir, _out_name("model.pkl", mt)), "wb") as f:
            pickle.dump(final_model, f)
        try:
            final_importance = final_model.feature_importances_.astype(float)
        except Exception:
            final_importance = np.zeros(num_features, dtype=float)
        test_proba = final_model.predict_proba(X_test)[:, 1]
    elif args.model == "l1":
        final_model = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=args.lr_c,
            class_weight="balanced",
            max_iter=100,
            random_state=args.random_state,
            n_jobs=-1,
        )
        final_model.fit(X_train, y_train)
        with open(os.path.join(args.out_dir, _out_name("model.pkl", mt)), "wb") as f:
            pickle.dump(final_model, f)
        if scaler is not None:
            with open(os.path.join(args.out_dir, _out_name("scaler.pkl", mt)), "wb") as f:
                pickle.dump(scaler, f)
        try:
            final_importance = np.abs(final_model.coef_).ravel().astype(float)
        except Exception:
            final_importance = np.zeros(num_features, dtype=float)
        test_proba = final_model.predict_proba(X_test)[:, 1]
    elif args.model == "enet":
        final_model = LogisticRegression(
            penalty="elasticnet",
            l1_ratio=args.enet_l1_ratio,
            solver="saga",
            C=args.lr_c,
            class_weight="balanced",
            max_iter=100,
            random_state=args.random_state,
            n_jobs=-1,
        )
        final_model.fit(X_train, y_train)
        with open(os.path.join(args.out_dir, _out_name("model.pkl", mt)), "wb") as f:
            pickle.dump(final_model, f)
        if scaler is not None:
            with open(os.path.join(args.out_dir, _out_name("scaler.pkl", mt)), "wb") as f:
                pickle.dump(scaler, f)
        try:
            final_importance = np.abs(final_model.coef_).ravel().astype(float)
        except Exception:
            final_importance = np.zeros(num_features, dtype=float)
        test_proba = final_model.predict_proba(X_test)[:, 1]

    pd.DataFrame({"feature": feature_names, "importance": final_importance}).to_csv(
        os.path.join(args.out_dir, "feature_importances.csv"), index=False
    )

    test_pred = (test_proba >= 0.5).astype(int)
    test_summary = compute_binary_metrics(y_test, test_proba)
    test_summary["train_size"] = int(len(train_idx))
    test_summary["test_size"] = int(len(test_idx))
    test_summary["train_chromosomes"] = int(train_df["chrom"].astype(str).nunique())
    test_summary["test_chromosomes"] = int(test_df["chrom"].astype(str).nunique())
    with open(os.path.join(args.out_dir, "test_summary.txt"), "w") as f:
        for k, v in test_summary.items():
            f.write(f"{k}\t{v}\n")

    pd.DataFrame(
        {"id": ids_test, "y_true": y_test.astype(int), "y_prob": test_proba, "y_pred": test_pred}
    ).to_csv(os.path.join(args.out_dir, "test_predictions.csv"), index=False)

    model_meta = {
        "model_type": args.model,
        "agg": args.agg,
        "use_dist_strand": bool(args.dist_strand_csv),
        "feature_meta": args.feature_meta,
        "feature_names": feature_names,
        "num_features": int(num_features),
        "id_col": args.id_col,
        "label_col": args.label_col,
        "threshold": 0.5,
        "xgb_rounds": args.xgb_rounds,
        "n_estimators": args.n_estimators,
        "lr_c": args.lr_c,
        "enet_l1_ratio": args.enet_l1_ratio,
        "xgb_gamma": args.xgb_gamma,
        "xgb_early_stopping_rounds": args.xgb_early_stopping_rounds,
        "xgb_scale_pos_weight": args.xgb_scale_pos_weight,
        "auto_tune": args.auto_tune,
        "best_params": tuned,
        "random_state": args.random_state,
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "tuning_folds_on_train": int(args.folds),
        "test_split_strategy": "chromosome_holdout_20_percent",
    }
    with open(os.path.join(args.out_dir, _out_name("model_meta.json", mt)), "w") as f:
        json.dump(model_meta, f, indent=2)

    print(f"Completed xQTL training/evaluation (chromosome-based holdout). Outputs at: {args.out_dir}")
    print(f"Final model and metadata saved under: {args.out_dir}")


if __name__ == "__main__":
    main()



