#!/usr/bin/env python3
import os
import sys
import argparse
import json
import pickle
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
from sklearn.model_selection import ParameterSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    import xgboost as xgb
    _HAVE_XGB = True
except Exception:
    xgb = None  # type: ignore
    _HAVE_XGB = False
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




def _parameter_candidates(grid: dict, budget: int, random_state: int) -> List[dict]:
    total = 1
    for values in grid.values():
        total *= len(values)
    n_iter = min(total, budget)
    return list(ParameterSampler(grid, n_iter=n_iter, random_state=random_state))


DEFAULT_XGB_ROUNDS = 100
DEFAULT_XGB_EARLY_STOPPING_ROUNDS = 30
DEFAULT_XGB_GAMMA = 0.0
DEFAULT_RF_N_ESTIMATORS = 100
DEFAULT_LR_C = 0.1
DEFAULT_ENET_L1_RATIO = 0.5
DEFAULT_SVM_KERNEL = "linear"
DEFAULT_SVM_GAMMA = "scale"


def _resolve_model_params(model: str, tuned: dict, y_train: Optional[np.ndarray] = None) -> dict:
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
        "n_estimators": int(tuned.get("n_estimators", DEFAULT_RF_N_ESTIMATORS)),
        "max_depth": tuned.get("max_depth", None),
        "min_samples_leaf": int(tuned.get("min_samples_leaf", 1)),
        "lr_c": float(tuned.get("lr_c", DEFAULT_LR_C)),
        "enet_l1_ratio": float(tuned.get("enet_l1_ratio", DEFAULT_ENET_L1_RATIO)),
        "svm_kernel": tuned.get("svm_kernel", DEFAULT_SVM_KERNEL),
        "svm_gamma": tuned.get("svm_gamma", DEFAULT_SVM_GAMMA),
    }
    if model == "xgb":
        pos = int(np.sum(y_train == 1)) if y_train is not None else 0
        neg = int(np.sum(y_train == 0)) if y_train is not None else 0
        params["xgb_scale_pos_weight"] = float(neg / max(pos, 1)) if y_train is not None else None
    return params


def _auto_tune_selected_model(args, train_df: pd.DataFrame, chrom_col: str, X: np.ndarray, y: np.ndarray) -> dict:
    if not getattr(args, "auto_tune", True):
        return {}
    base_params = _resolve_model_params(args.model, {}, y)
    tune_folds = get_chromosome_based_kfold(train_df, chrom_col, args.folds)
    if not tune_folds:
        print("WARNING: auto tuning skipped because train-split chromosome folds could not be built.", file=sys.stderr)
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
        grid = {
            "lr_c": [1e-2, 1e-1, 1],
            "svm_kernel": ["linear"],
        }
    samples = _parameter_candidates(grid, budget=20, random_state=args.random_state)
    for cand in samples:
        aucs = []
        for tr_idx, te_idx in tune_folds:
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
                    "scale_pos_weight": base_params["xgb_scale_pos_weight"],
                }
                m = xgb.train(
                    p,
                    dtr,
                    num_boost_round=int(cand["xgb_rounds"]),
                    evals=[(dte, "valid")],
                    early_stopping_rounds=DEFAULT_XGB_EARLY_STOPPING_ROUNDS,
                    verbose_eval=False,
                )
                proba = m.predict(dte)
            elif args.model == "rf":
                m = RandomForestClassifier(
                    n_estimators=int(cand["n_estimators"]),
                    max_depth=cand["max_depth"],
                    min_samples_leaf=int(cand["min_samples_leaf"]),
                    max_features="log2",
                    class_weight="balanced",
                    n_jobs=args.n_jobs,
                    random_state=args.random_state,
                )
                m.fit(X_tr, y_tr)
                proba = m.predict_proba(X_te)[:, 1]
            elif args.model == "l1":
                m = LogisticRegression(penalty="l1", solver="saga", C=float(cand["lr_c"]), class_weight="balanced", max_iter=100, random_state=args.random_state, n_jobs=args.n_jobs)
                m.fit(X_tr, y_tr)
                proba = m.predict_proba(X_te)[:, 1]
            elif args.model == "enet":
                m = LogisticRegression(penalty="elasticnet", l1_ratio=float(cand["enet_l1_ratio"]), solver="saga", C=float(cand["lr_c"]), class_weight="balanced", max_iter=100, random_state=args.random_state, n_jobs=args.n_jobs)
                m.fit(X_tr, y_tr)
                proba = m.predict_proba(X_te)[:, 1]
            else:
                kw = {"C": float(cand["lr_c"]), "kernel": cand["svm_kernel"], "probability": True, "class_weight": "balanced", "random_state": args.random_state}
                if cand["svm_kernel"] != "linear":
                    kw["gamma"] = cand["svm_gamma"]
                kw["max_iter"] = 100
                m = SVC(**kw)
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
    if best:
        print(f"Auto-tuning on the train split with chromosome-based internal {args.folds}-fold CV evaluated {len(samples)} candidates; best AUC={best_auc:.6f}, best_params={best}")
    return best


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Classification models (RF/XGB/L1/ElasticNet/SVM) + cross-validation on Enformer outputs; optional save/predict workflow."
    )
    p.add_argument(
        "--mode",
        type=str,
        choices=["train", "predict"],
        default="train",
        help="train: CV + save final model; predict: score a new Enformer CSV.",
    )
    p.add_argument("--enformer_csv", type=str, required=True,
                   help="CSV produced by score_vcf_enformer.py (contains tracks and variant identifiers).")
    p.add_argument(
        "--labels_csv",
        type=str,
        default=None,
        help="CSV with variant id and binary label (required for --mode train).",
    )
    p.add_argument("--id_col", type=str, default="id",
                   help="Variant id column name (must exist in both inputs).")
    p.add_argument("--label_col", type=str, default="label",
                   help="Column name for binary label (0/1).")

    p.add_argument("--folds", type=int, default=5, help="Number of chromosome-based folds used only for auto-tuning on the train split (default: 5).")
    p.add_argument("--n_jobs", type=int, default=-1, help="Parallel jobs for RandomForest (default: -1).")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory.")
    p.add_argument("--model", type=str, choices=["rf", "xgb", "l1", "enet", "svm"], default="rf",
                   help="Model to use: rf, xgb, l1 (L1 logistic), enet (elastic-net logistic), svm (SVC with linear/RBF kernel probability).")
    p.add_argument("--auto_tune", dest="auto_tune", action="store_true", default=True, help="Enable train-split auto tuning with chromosome-based internal folds defined by --folds (default: on).")
    p.add_argument("--no_auto_tune", dest="auto_tune", action="store_false", help="Disable train-split auto tuning and use built-in model defaults.")
    p.add_argument("--random_state", type=int, default=0, help="Random seed base (used by CV/model).")
    p.add_argument(
        "--chrom_col",
        type=str,
        default="chrom",
        help="Chromosome column name for chromosome-based folds.",
    )
    p.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Directory with model_meta_{mt}.json, model_{mt}.pkl|json, scaler_{mt}.pkl (xgb/l1/enet/svm). --model must match training.",
    )
    p.add_argument(
        "--predict_out",
        type=str,
        default=None,
        help="Prediction CSV basename under --out_dir; default: predictions_{model}.csv.",
    )
    return p.parse_args()


def ensure_out_dir(path: str) -> None:
    """Create output directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def _out_name(filename: str, model_tag: str) -> str:
    base, ext = os.path.splitext(filename)
    return f"{base}_{model_tag}{ext}" if ext else f"{filename}_{model_tag}"


def detect_enformer_features(df: pd.DataFrame):
    """Detect feature columns in Enformer outputs (only full mode is kept)."""
    meta_cols = {"chrom", "pos", "id", "ref", "alt","label"}
    cols = list(df.columns)
    track_cols = [c for c in cols if c.startswith("track_")]
    indexed_cols = [c for c in cols if ("_" in c and c.split("_")[0].isdigit())]

    if track_cols:
        return track_cols
    if indexed_cols:
        return indexed_cols
    return [c for c in cols if c not in meta_cols]


def align_enformer_features(df: pd.DataFrame, feature_cols: List[str], id_col: str) -> Tuple[np.ndarray, List[str]]:
    """Build X in training column order; missing features filled with 0."""
    if id_col not in df.columns:
        print(f"ERROR: id_col '{id_col}' not found in Enformer CSV columns.", file=sys.stderr)
        sys.exit(1)
    ids = df[id_col].astype(str).tolist()
    n = len(df)
    X = np.zeros((n, len(feature_cols)), dtype=np.float32)
    for j, c in enumerate(feature_cols):
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            X[:, j] = s.to_numpy(dtype=np.float32)
    return X, ids


def get_chromosome_based_kfold(
    df: pd.DataFrame, chrom_col: str, n_folds: int
) -> list:
    """
    Assign chromosomes to folds to balance sample counts, then return a list of
    (train_idx, test_idx) tuples. Folds with no samples in either split are skipped.
    """
    if chrom_col not in df.columns:
        raise ValueError(f"Chromosome column '{chrom_col}' not found in input DataFrame.")

    # Count samples per chromosome and sort chromosomes by descending sample count
    chrom_counts = df[chrom_col].value_counts().to_dict()
    chrom_samples = sorted(chrom_counts.items(), key=lambda x: x[1], reverse=True)

    # Greedy balancing: assign next heaviest chromosome to the fold with least samples
    fold_assignments = {i: [] for i in range(n_folds)}
    fold_sample_counts = [0] * n_folds
    for chrom, count in chrom_samples:
        min_fold = fold_sample_counts.index(min(fold_sample_counts))
        fold_assignments[min_fold].append(chrom)
        fold_sample_counts[min_fold] += count

    # Build train/test indices per fold
    folds = []
    for fold in range(n_folds):
        test_chroms = fold_assignments[fold]
        if len(test_chroms) == 0:
            continue
        test_mask = df[chrom_col].isin(test_chroms).to_numpy()
        train_mask = ~test_mask
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(train_mask)[0]
        if len(test_idx) > 0 and len(train_idx) > 0:
            folds.append((train_idx, test_idx))
    return folds


def get_chromosome_train_test_split(df: pd.DataFrame, chrom_col: str) -> Tuple[np.ndarray, np.ndarray]:
    holdout_folds = get_chromosome_based_kfold(df, chrom_col=chrom_col, n_folds=5)
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


def run_analysis(enf: pd.DataFrame, labels: pd.DataFrame, args: argparse.Namespace) -> None:
    """Main analysis pipeline: merge data, extract features, run CV, and save results."""
    if args.model == "xgb" and not _HAVE_XGB:
        print("ERROR: --model xgb requested but xgboost is not available.", file=sys.stderr)
        sys.exit(1)

    if args.id_col not in enf.columns or args.id_col not in labels.columns:
        print(f"ERROR: id_col '{args.id_col}' must exist in both inputs", file=sys.stderr)
        sys.exit(1)

    merged = pd.merge(labels, enf, on=args.id_col, how="inner")
    if merged.empty:
        print("ERROR: After merging labels and Enformer CSV by id_col, no rows remain.", file=sys.stderr)
        sys.exit(1)

    features = detect_enformer_features(merged)
    if args.label_col not in merged.columns:
        print("ERROR: labels_csv must include label column.", file=sys.stderr)
        sys.exit(1)

    mt = args.model
    X = merged[features].to_numpy(dtype=float)
    y = merged[args.label_col].to_numpy(dtype=int)
    ids = merged[args.id_col].astype(str).to_numpy()

    # Build chromosome-based holdout and train-only CV folds
    if args.chrom_col not in merged.columns:
        print(
            f"ERROR: chrom_col '{args.chrom_col}' not found after merging; available columns: {list(merged.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        train_idx, test_idx = get_chromosome_train_test_split(merged, args.chrom_col)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    train_df = merged.iloc[train_idx].reset_index(drop=True)
    test_df = merged.iloc[test_idx].reset_index(drop=True)
    X_train = X[train_idx]
    y_train = y[train_idx]
    ids_train = ids[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    ids_test = ids[test_idx]

    scaler = None
    if args.model in {"xgb", "l1", "enet", "svm"}:
        ensure_out_dir(args.out_dir)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        np.savetxt(os.path.join(args.out_dir, _out_name("scaler_mean.txt", mt)), scaler.mean_, fmt="%.6f")
        np.savetxt(os.path.join(args.out_dir, _out_name("scaler_scale.txt", mt)), scaler.scale_, fmt="%.6f")

    tuned = _auto_tune_selected_model(args, train_df, args.chrom_col, X_train, y_train)
    model_params = _resolve_model_params(args.model, tuned, y_train)
    ensure_out_dir(args.out_dir)
    num_features = X_train.shape[1]

    # Final model on the full training split only (for --mode predict, held-out
    # test evaluation, and final-model feature importances).
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
    elif args.model == "rf":
        final_model = RandomForestClassifier(
            n_estimators=model_params["n_estimators"],
            max_features="log2",
            class_weight="balanced",
            n_jobs=args.n_jobs,
            random_state=args.random_state,
        )
        final_model.fit(X_train, y_train)
        with open(os.path.join(args.out_dir, _out_name("model.pkl", mt)), "wb") as f:
            pickle.dump(final_model, f)
        test_proba = final_model.predict_proba(X_test)[:, 1]
    elif args.model == "l1":
        final_model = LogisticRegression(penalty="l1", solver="saga", C=model_params["lr_c"], class_weight="balanced", max_iter=100, random_state=args.random_state, n_jobs=args.n_jobs)
        final_model.fit(X_train, y_train)
        with open(os.path.join(args.out_dir, _out_name("model.pkl", mt)), "wb") as f:
            pickle.dump(final_model, f)
        if scaler is not None:
            with open(os.path.join(args.out_dir, _out_name("scaler.pkl", mt)), "wb") as f:
                pickle.dump(scaler, f)
        test_proba = final_model.predict_proba(X_test)[:, 1]
    elif args.model == "enet":
        final_model = LogisticRegression(penalty="elasticnet", l1_ratio=model_params["enet_l1_ratio"], solver="saga", C=model_params["lr_c"], class_weight="balanced", max_iter=100, random_state=args.random_state, n_jobs=args.n_jobs)
        final_model.fit(X_train, y_train)
        with open(os.path.join(args.out_dir, _out_name("model.pkl", mt)), "wb") as f:
            pickle.dump(final_model, f)
        if scaler is not None:
            with open(os.path.join(args.out_dir, _out_name("scaler.pkl", mt)), "wb") as f:
                pickle.dump(scaler, f)
        test_proba = final_model.predict_proba(X_test)[:, 1]
    else:
        svm_kwargs = {
            "C": model_params["lr_c"],
            "kernel": model_params["svm_kernel"],
            "probability": True,
            "class_weight": "balanced",
            "random_state": args.random_state,
        }
        if model_params["svm_kernel"] != "linear":
            svm_kwargs["gamma"] = model_params["svm_gamma"]
        final_model = SVC(**svm_kwargs)
        final_model.fit(X_train, y_train)
        with open(os.path.join(args.out_dir, _out_name("model.pkl", mt)), "wb") as f:
            pickle.dump(final_model, f)
        if scaler is not None:
            with open(os.path.join(args.out_dir, _out_name("scaler.pkl", mt)), "wb") as f:
                pickle.dump(scaler, f)
        test_proba = final_model.predict_proba(X_test)[:, 1]

    final_importance = get_model_feature_importance(final_model, args.model, num_features)
    pd.DataFrame({"feature": features, "importance": final_importance}).to_csv(
        os.path.join(args.out_dir, "feature_importances.csv"), index=False
    )

    test_pred = (test_proba >= 0.5).astype(int)
    test_summary = compute_binary_metrics(y_test, test_proba)
    test_summary["train_size"] = int(len(train_idx))
    test_summary["test_size"] = int(len(test_idx))
    test_summary["train_chromosomes"] = int(train_df[args.chrom_col].astype(str).nunique())
    test_summary["test_chromosomes"] = int(test_df[args.chrom_col].astype(str).nunique())
    with open(os.path.join(args.out_dir, "test_summary.txt"), "w") as f:
        for k, v in test_summary.items():
            f.write(f"{k}\t{v}\n")

    test_pred_df = pd.DataFrame(
        {"id": ids_test, "y_true": y_test.astype(int), "y_prob": test_proba, "y_pred": test_pred}
    )
    test_pred_df.to_csv(os.path.join(args.out_dir, "test_predictions.csv"), index=False)

    model_meta = {
        "model_type": args.model,
        "feature_cols": features,
        "id_col": args.id_col,
        "label_col": args.label_col,
        "threshold": 0.5,
        "lr_c": model_params["lr_c"],
        "enet_l1_ratio": model_params["enet_l1_ratio"],
        "svm_kernel": model_params["svm_kernel"],
        "svm_gamma": model_params["svm_gamma"],
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

    print(f"Completed analysis with chromosome holdout test evaluation. Outputs at: {args.out_dir}")
    print(f"Final model and metadata saved under: {args.out_dir}")


def run_predict(args: argparse.Namespace) -> None:
    """Load saved model and score rows in --enformer_csv (same id_col / features as training)."""
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

    id_col = meta["id_col"]
    trained_features: List[str] = meta["feature_cols"]
    enf = pd.read_csv(args.enformer_csv)
    X, ids = align_enformer_features(enf, trained_features, id_col)

    if model_type == "xgb":
        scaler_path = os.path.join(args.model_dir, _out_name("scaler.pkl", model_type))
        if not os.path.isfile(scaler_path):
            print(f"ERROR: missing {scaler_path}", file=sys.stderr)
            sys.exit(1)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        X = scaler.transform(X)
        model_path = os.path.join(args.model_dir, _out_name("model.json", model_type))
        if not os.path.isfile(model_path):
            print(f"ERROR: missing {model_path}", file=sys.stderr)
            sys.exit(1)
        booster = xgb.Booster()
        booster.load_model(model_path)
        proba = booster.predict(xgb.DMatrix(X))
    else:
        if model_type in {"l1", "enet", "svm"}:
            scaler_path = os.path.join(args.model_dir, _out_name("scaler.pkl", model_type))
            if not os.path.isfile(scaler_path):
                print(f"ERROR: missing {scaler_path}", file=sys.stderr)
                sys.exit(1)
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            X = scaler.transform(X)
        model_path = os.path.join(args.model_dir, _out_name("model.pkl", model_type))
        if not os.path.isfile(model_path):
            print(f"ERROR: missing {model_path}", file=sys.stderr)
            sys.exit(1)
        with open(model_path, "rb") as f:
            clf = pickle.load(f)
        proba = clf.predict_proba(X)[:, 1]

    thr = float(meta.get("threshold", 0.5))
    pred = (proba >= thr).astype(int)
    pred_df = pd.DataFrame({"id": ids, "y_prob": proba, "y_pred": pred})
    pred_bn = args.predict_out if args.predict_out else _out_name("predictions.csv", model_type)
    pred_path = os.path.join(args.out_dir, pred_bn)
    pred_df.to_csv(pred_path, index=False)
    print(f"Prediction completed. Results written to: {pred_path}")


def main():
    args = parse_args()
    if args.model == "xgb" and not _HAVE_XGB:
        print("ERROR: --model xgb requested but xgboost is not available.", file=sys.stderr)
        sys.exit(1)
    if args.mode == "train" and not args.labels_csv:
        print("ERROR: --mode train requires --labels_csv.", file=sys.stderr)
        sys.exit(1)

    ensure_out_dir(args.out_dir)

    if args.mode == "predict":
        run_predict(args)
        return

    enf = pd.read_csv(args.enformer_csv)
    labels = pd.read_csv(args.labels_csv)
    run_analysis(enf, labels, args)


if __name__ == "__main__":
    main()
