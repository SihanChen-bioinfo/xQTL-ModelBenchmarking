#!/usr/bin/env python3
import os
import sys
import json
import pickle
import argparse
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.model_selection import ParameterSampler
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    _HAVE_XGB = True
except Exception:
    xgb = None  # type: ignore
    _HAVE_XGB = False


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
    "lr_c": 0.1,
    "enet_l1_ratio": 0.5,
}


def _initialize_model_defaults(args: argparse.Namespace, y_train: Optional[np.ndarray] = None) -> None:
    for key, value in MODEL_DEFAULTS.items():
        setattr(args, key, value)
    if y_train is not None:
        pos = int(np.sum(y_train == 1))
        neg = int(np.sum(y_train == 0))
        args.xgb_scale_pos_weight = float(neg / max(pos, 1))


def _out_name(filename: str, model_tag: str) -> str:
    base, ext = os.path.splitext(filename)
    return f"{base}_{model_tag}{ext}" if ext else f"{filename}_{model_tag}"




def _sample_tuning_candidates(grid: dict, max_trials: int, seed: int) -> List[dict]:
    total = 1
    for values in grid.values():
        total *= len(values)
    if total <= max_trials:
        return list(ParameterSampler(grid, n_iter=total, random_state=seed))
    return list(ParameterSampler(grid, n_iter=max_trials, random_state=seed))


def _auto_tune_selected_model(args, X: np.ndarray, y: np.ndarray, train_df: pd.DataFrame) -> dict:
    if not getattr(args, "auto_tune", True):
        return {}
    counts = np.bincount(y.astype(int))
    if counts.size < 2 or np.min(counts) < 2:
        print("WARNING: auto tuning skipped due to insufficient class samples.", file=sys.stderr)
        return {}
    cv = _chrom_folds(train_df, n_folds=args.folds, seed=args.random_state)
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
                m = RandomForestClassifier(n_estimators=int(cand["n_estimators"]), max_depth=cand["max_depth"], min_samples_leaf=int(cand["min_samples_leaf"]), max_features="log2", class_weight="balanced", n_jobs=args.n_jobs, random_state=args.random_state)
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
    p = argparse.ArgumentParser(description="GWAVA classifiers with chromosome-based train/test split and train-only chromosome tuning CV. Model-specific hyperparameters come from internal defaults plus auto-tune outputs.")
    p.add_argument("--mode", choices=["train", "predict"], default="train")
    p.add_argument("--model", choices=["rf", "xgb", "l1", "enet"], default="xgb")
    p.add_argument("--gwava_dir", type=str, default=os.getenv("GWAVA_DIR", "."))
    p.add_argument("--data_subdir", type=str, default="xQTL_data")
    p.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., bQTL).")
    p.add_argument("--out_dir", type=str, default=None, help="Output directory. All output files are written directly here.")
    p.add_argument("--model_dir", type=str, default=None, help="Model directory for predict (default: --out_dir).")
    p.add_argument("--predict_out", type=str, default=None)
    p.add_argument("--folds", type=int, default=5, help="Number of chromosome-based folds used for auto-tuning on the training split.")
    p.add_argument("--random_state", type=int, default=0)
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--auto_tune", dest="auto_tune", action="store_true", default=True, help="Enable automatic hyperparameter tuning on the train split using internal chromosome-based folds from --folds.")
    p.add_argument("--no_auto_tune", dest="auto_tune", action="store_false", help="Disable automatic tuning and use internal model defaults for training.")
    return p.parse_args()


def _read_any(stem: str) -> Optional[pd.DataFrame]:
    for ext in (".pandas", ".pkl", ".h5"):
        p = stem + ext
        if os.path.exists(p):
            if ext in (".pandas", ".pkl"):
                return pd.read_pickle(p)
            return pd.read_hdf(p, key="data")
    return None


def _load_dataset(ds_name: str, data_dir: str) -> pd.DataFrame:
    c1 = _read_any(os.path.join(data_dir, f"1KG_hg19.filter.merged.{ds_name}_control_set.MAF0.01.bed.annotate"))
    c2 = _read_any(os.path.join(data_dir, f"1KG_hg19.filter.merged.{ds_name}.MAF0.01.bed.annotate"))
    if c1 is not None and c2 is not None:
        return pd.concat([c1, c2], axis=0, ignore_index=True, sort=False)

    raise FileNotFoundError(f"Could not locate GWAVA files for dataset '{ds_name}' in {data_dir}")


def _detect_features(df: pd.DataFrame) -> List[str]:
    drop_cols = {"start", "end", "chr", "cls", "id", "ID"}
    return [c for c in df.columns if c not in drop_cols]


def _build_ids(df: pd.DataFrame) -> np.ndarray:
    if "chr" in df.columns and "end" in df.columns:
        return (df["chr"].astype(str) + "_" + df["end"].astype(str)).values
    if "id" in df.columns:
        return df["id"].astype(str).values
    if "ID" in df.columns:
        return df["ID"].astype(str).values
    return np.arange(len(df)).astype(str)


def _chrom_folds(df: pd.DataFrame, n_folds: int, seed: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    vc = df["chr"].astype(str).value_counts().to_dict()
    items = list(vc.items())
    rng = np.random.RandomState(seed)
    rng.shuffle(items)
    items.sort(key=lambda x: x[1], reverse=True)
    fold_assign = {i: [] for i in range(n_folds)}
    fold_sizes = [0] * n_folds
    for ch, cnt in items:
        k = fold_sizes.index(min(fold_sizes))
        fold_assign[k].append(ch)
        fold_sizes[k] += cnt
    folds = []
    chr_arr = df["chr"].astype(str).values
    for k in range(n_folds):
        mask = np.isin(chr_arr, fold_assign[k])
        te = np.where(mask)[0]
        tr = np.where(~mask)[0]
        if len(te) > 0 and len(tr) > 0:
            folds.append((tr, te))
    return folds


def _chrom_train_test_split(df: pd.DataFrame, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    holdout_folds = _chrom_folds(df, n_folds=5, seed=seed)
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


def _make_clf(args: argparse.Namespace, seed: int):
    if args.model == "rf":
        return RandomForestClassifier(n_estimators=args.n_estimators, max_depth=getattr(args, "max_depth", None), min_samples_leaf=int(getattr(args, "min_samples_leaf", 1)), max_features="log2", class_weight="balanced", n_jobs=args.n_jobs, random_state=seed)
    if args.model == "l1":
        return LogisticRegression(penalty="l1", solver="saga", C=args.lr_c, class_weight="balanced", max_iter=100, random_state=seed, n_jobs=args.n_jobs)
    if args.model == "enet":
        return LogisticRegression(penalty="elasticnet", l1_ratio=args.enet_l1_ratio, solver="saga", C=args.lr_c, class_weight="balanced", max_iter=100, random_state=seed, n_jobs=args.n_jobs)
    return None


def _train_one_dataset(args: argparse.Namespace, ds_name: str, data_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = _load_dataset(ds_name, data_dir)
    if "cls" not in df.columns or "chr" not in df.columns:
        raise ValueError(f"{ds_name}: requires columns 'cls' and 'chr'")
    features = _detect_features(df)
    X = df[features].astype(float).fillna(0.0).values
    y = df["cls"].astype(int).values
    ids = _build_ids(df)
    mt = args.model

    try:
        train_idx, test_idx = _chrom_train_test_split(df, seed=args.random_state)
    except ValueError as e:
        raise ValueError(f"{ds_name}: {e}") from e

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    ids_test = ids[test_idx]

    scaler = None
    if args.model in {"xgb", "l1", "enet"}:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        np.savetxt(os.path.join(out_dir, _out_name("scaler_mean.txt", mt)), scaler.mean_, fmt="%.6f")
        np.savetxt(os.path.join(out_dir, _out_name("scaler_scale.txt", mt)), scaler.scale_, fmt="%.6f")

    _initialize_model_defaults(args, y_train=y_train)
    tuned = _auto_tune_selected_model(args, X_train, y_train, train_df)
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
        dall = xgb.DMatrix(X_train, label=y_train)
        fm = xgb.train(params, dall, num_boost_round=args.xgb_rounds)
        fm.save_model(os.path.join(out_dir, _out_name("model.json", mt)))
        with open(os.path.join(out_dir, _out_name("scaler.pkl", mt)), "wb") as f:
            pickle.dump(scaler, f)
        final_importance = np.zeros(X_train.shape[1], dtype=float)
        try:
            for k, v in fm.get_score(importance_type="gain").items():
                if k.startswith("f"):
                    j = int(k[1:])
                    if 0 <= j < X_train.shape[1]:
                        final_importance[j] = float(v)
        except Exception:
            pass
        test_proba = fm.predict(xgb.DMatrix(X_test))
    else:
        fm = _make_clf(args, args.random_state)
        fm.fit(X_train, y_train)
        with open(os.path.join(out_dir, _out_name("model.pkl", mt)), "wb") as f:
            pickle.dump(fm, f)
        if scaler is not None:
            with open(os.path.join(out_dir, _out_name("scaler.pkl", mt)), "wb") as f:
                pickle.dump(scaler, f)
        try:
            if hasattr(fm, "feature_importances_"):
                final_importance = fm.feature_importances_.astype(float)
            elif hasattr(fm, "coef_"):
                final_importance = np.abs(fm.coef_).ravel().astype(float)
            else:
                final_importance = np.zeros(X_train.shape[1], dtype=float)
        except Exception:
            final_importance = np.zeros(X_train.shape[1], dtype=float)
        test_proba = fm.predict_proba(X_test)[:, 1]

    with open(os.path.join(out_dir, _out_name("model_meta.json", mt)), "w") as f:
        json.dump({
            "dataset": ds_name,
            "model_type": args.model,
            "feature_cols": features,
            "num_features": int(X_train.shape[1]),
            "id_col": "id",
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
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "tuning_folds_on_train": int(args.folds),
            "test_split_strategy": "chromosome_holdout_20_percent",
        }, f, indent=2)

    pd.DataFrame({"feature": features, "importance": final_importance}).to_csv(
        os.path.join(out_dir, "feature_importances.csv"), index=False
    )

    test_pred = (test_proba >= 0.5).astype(int)
    test_summary = compute_binary_metrics(y_test, test_proba)
    test_summary["train_size"] = int(len(train_idx))
    test_summary["test_size"] = int(len(test_idx))
    test_summary["train_chromosomes"] = int(train_df["chr"].astype(str).nunique())
    test_summary["test_chromosomes"] = int(test_df["chr"].astype(str).nunique())
    with open(os.path.join(out_dir, "test_summary.txt"), "w") as f:
        for k, v in test_summary.items():
            f.write(f"{k}\t{v}\n")

    pd.DataFrame(
        {"id": ids_test.astype(str), "y_true": y_test.astype(int), "y_prob": test_proba, "y_pred": test_pred}
    ).to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)


def _predict_one_dataset(args: argparse.Namespace, ds_name: str, data_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    model_dir = args.model_dir if args.model_dir else out_dir
    meta_path = os.path.join(model_dir, _out_name("model_meta.json", args.model))
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing metadata: {meta_path}")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    if meta.get("model_type") != args.model:
        raise ValueError(f"--model {args.model} mismatch meta model_type {meta.get('model_type')}")

    df = _load_dataset(ds_name, data_dir)
    X = np.zeros((len(df), len(meta["feature_cols"])), dtype=np.float64)
    for j, c in enumerate(meta["feature_cols"]):
        if c in df.columns:
            X[:, j] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    ids = _build_ids(df)

    if args.model == "xgb":
        if not _HAVE_XGB:
            raise RuntimeError("xgboost not installed")
        scaler_path = os.path.join(model_dir, _out_name("scaler.pkl", args.model))
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        X = scaler.transform(X)
        booster = xgb.Booster()
        booster.load_model(os.path.join(model_dir, _out_name("model.json", args.model)))
        proba = booster.predict(xgb.DMatrix(X))
    else:
        if args.model in {"l1", "enet"}:
            scaler_path = os.path.join(model_dir, _out_name("scaler.pkl", args.model))
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            X = scaler.transform(X)
        with open(os.path.join(model_dir, _out_name("model.pkl", args.model)), "rb") as f:
            clf = pickle.load(f)
        proba = clf.predict_proba(X)[:, 1]

    thr = float(meta.get("threshold", 0.5))
    pred = (proba >= thr).astype(int)
    pred_name = args.predict_out if args.predict_out else _out_name("predictions.csv", args.model)
    pd.DataFrame({"id": ids, "y_prob": proba, "y_pred": pred}).to_csv(os.path.join(out_dir, pred_name), index=False)


def main() -> None:
    args = parse_args()
    if args.model == "xgb" and not _HAVE_XGB:
        print("ERROR: xgboost is not available", file=sys.stderr)
        sys.exit(1)

    data_dir = os.path.join(args.gwava_dir, args.data_subdir)
    out_dir = args.out_dir if args.out_dir else os.path.join(args.gwava_dir, "cross_validation_result")
    os.makedirs(out_dir, exist_ok=True)

    ds = args.dataset
    if args.mode == "train":
        _train_one_dataset(args, ds, data_dir, out_dir)
    else:
        _predict_one_dataset(args, ds, data_dir, out_dir)
    print(f"Completed {args.mode} for {ds}")


if __name__ == "__main__":
    main()
