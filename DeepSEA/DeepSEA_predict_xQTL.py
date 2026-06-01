import sys
import os
import json
import pickle
import argparse
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    _HAVE_XGB = True
except Exception:
    xgb = None  # type: ignore
    _HAVE_XGB = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)
from sklearn.preprocessing import StandardScaler


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


def _auto_tune_selected_model(args, X: np.ndarray, y: np.ndarray, chromosomes: List[str]) -> dict:
    if not getattr(args, "auto_tune", True):
        return {}
    counts = np.bincount(y.astype(int))
    if counts.size < 2 or np.min(counts) < 2:
        print("WARNING: auto tuning skipped due to insufficient class samples.", file=sys.stderr)
        return {}
    cv = get_chromosome_folds(chromosomes, n_folds=args.folds, seed=args.random_state)
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


def parse_args():
    parser = argparse.ArgumentParser(description="DeepSEA classifiers (XGB/RF/L1/ElasticNet) with chromosome-based train/test split and train-only chromosome tuning CV; train or predict. Model-specific hyperparameters come from internal defaults plus auto-tune outputs.")
    parser.add_argument("--features_csv", type=str, required=True, help="Training/predict features CSV (chr,pos,name,ref,alt, feats...).")
    parser.add_argument(
        "--labels_csv",
        type=str,
        default=None,
        help="Labels CSV with columns name,label (required for --mode train; ignored for predict).",
    )
    parser.add_argument(
        "--xqtl_type",
        type=str,
        default="deepsea_xqtl",
        help="Run tag used in default output folder name if --out_dir not set.",
    )
    parser.add_argument("--mode", type=str, choices=["train", "predict"], default="train")
    parser.add_argument("--model", type=str, choices=["xgb", "rf", "l1", "enet"], default="xgb")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--auto_tune", dest="auto_tune", action="store_true", default=True, help="Enable automatic hyperparameter tuning on the train split using internal chromosome-based folds from --folds.")
    parser.add_argument("--no_auto_tune", dest="auto_tune", action="store_false", help="Disable automatic tuning and use internal model defaults for training.")
    parser.add_argument("--folds", type=int, default=5, help="Number of chromosome-based folds used for auto-tuning on the training split.")
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: {xqtl_type}_cross_validation_result).",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Saved model directory (for --mode predict): model_meta_{mt}.json, model_{mt}.pkl|json, scaler_{mt}.pkl (xgb/l1/enet). Use --model matching training.",
    )
    parser.add_argument(
        "--predict_out",
        type=str,
        default=None,
        help="Prediction CSV basename under --out_dir; default: predictions_{model}.csv.",
    )
    return parser.parse_args()


def load_features_and_merge_labels(features_path: str, labels_path: Optional[str], train_mode: bool):
    """Returns merged, X, y (empty y if predict), chromosomes, sample_ids, feature_cols, feature_names."""
    features_df = pd.read_csv(features_path, sep=None, engine="python")
    required_cols = {"chr", "pos", "name", "ref", "alt"}
    if not required_cols.issubset(set(features_df.columns)):
        print(f"ERROR: features file must contain columns: {sorted(list(required_cols))}")
        sys.exit(1)
    fixed_cols = ["chr", "pos", "name", "ref", "alt"]
    feature_cols = [c for c in features_df.columns if c not in fixed_cols]
    if len(feature_cols) == 0:
        print("ERROR: No feature columns detected after chr,pos,name,ref,alt")
        sys.exit(1)

    if train_mode:
        if not labels_path:
            print("ERROR: --mode train requires --labels_csv.", file=sys.stderr)
            sys.exit(1)
        labels = pd.read_csv(labels_path, sep=None, engine="python")
        if not {"name", "label"}.issubset(set(labels.columns)):
            print("ERROR: labels file must have header with columns: name,label")
            sys.exit(1)
        labels = labels[["name", "label"]].copy()
        merged = pd.merge(features_df, labels, on="name", how="inner")
        if merged.empty:
            print("ERROR: After merging features and labels by name/id, no rows remain.")
            sys.exit(1)
        if merged.shape[0] != features_df.shape[0]:
            print(
                f"Warning: {features_df.shape[0] - merged.shape[0]} samples in features have no matching label and will be dropped."
            )
        y = merged["label"].astype(int).values
    else:
        merged = features_df
        y = np.array([], dtype=int)

    X = merged[feature_cols].values
    chromosomes = merged["chr"].astype(str).tolist()
    sample_ids = merged["name"].astype(str).values
    feature_names = feature_cols[:]
    return merged, X, y, chromosomes, sample_ids, feature_cols, feature_names


def align_feature_matrix(features_df: pd.DataFrame, trained_cols: List[str]) -> np.ndarray:
    n = len(features_df)
    X = np.zeros((n, len(trained_cols)), dtype=np.float64)
    for j, c in enumerate(trained_cols):
        if c in features_df.columns:
            X[:, j] = pd.to_numeric(features_df[c], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    return X


def run_predict(args, output_dir: str) -> None:
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

    trained_cols = meta["feature_cols"]
    features_df = pd.read_csv(args.features_csv, sep=None, engine="python")
    required_cols = {"chr", "pos", "name", "ref", "alt"}
    if not required_cols.issubset(set(features_df.columns)):
        print(f"ERROR: features file must contain columns: {sorted(list(required_cols))}")
        sys.exit(1)

    X = align_feature_matrix(features_df, trained_cols)
    if X.shape[1] != int(meta.get("num_features", X.shape[1])):
        print(
            f"ERROR: feature count mismatch: model expects {meta.get('num_features')}, got {X.shape[1]}.",
            file=sys.stderr,
        )
        sys.exit(1)

    ids = features_df["name"].astype(str).values

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
        if model_type in {"l1", "enet"}:
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
    os.makedirs(output_dir, exist_ok=True)
    pred_bn = args.predict_out if args.predict_out else _out_name("predictions.csv", model_type)
    pred_path = os.path.join(output_dir, pred_bn)
    pd.DataFrame({"id": ids, "y_prob": proba, "y_pred": pred}).to_csv(pred_path, index=False)
    print(f"Prediction completed. Results written to: {pred_path}")


def get_chromosome_folds(chromosomes: List[str], n_folds: int, seed: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    chrom_series = pd.Series(chromosomes, dtype=str)
    chrom_counts = chrom_series.value_counts().to_dict()
    chrom_samples = list(chrom_counts.items())
    rng = np.random.RandomState(seed)
    rng.shuffle(chrom_samples)
    chrom_samples.sort(key=lambda x: x[1], reverse=True)
    fold_assignments = {i: [] for i in range(n_folds)}
    fold_sample_counts = [0] * n_folds

    for chrom, sample_count in chrom_samples:
        min_fold = fold_sample_counts.index(min(fold_sample_counts))
        fold_assignments[min_fold].append(chrom)
        fold_sample_counts[min_fold] += sample_count

    folds = []
    chrom_arr = chrom_series.to_numpy(dtype=str)
    for fold in range(n_folds):
        test_chroms = fold_assignments[fold]
        if not test_chroms:
            continue
        mask = np.isin(chrom_arr, test_chroms)
        test_idx = np.where(mask)[0]
        train_idx = np.where(~mask)[0]
        if len(train_idx) > 0 and len(test_idx) > 0:
            folds.append((train_idx, test_idx))
    return folds


def get_chromosome_train_test_split(chromosomes: List[str], seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    holdout_folds = get_chromosome_folds(chromosomes, n_folds=5, seed=seed)
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


def main():
    args = parse_args()
    if args.mode == "train" and args.model == "xgb" and not _HAVE_XGB:
        print("ERROR: --model xgb requested but xgboost is not available.", file=sys.stderr)
        sys.exit(1)

    output_dir = args.out_dir if args.out_dir else (str(args.xqtl_type) + "_cross_validation_result")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    if args.mode == "predict":
        run_predict(args, output_dir)
        return

    features_path = args.features_csv
    labels_path = args.labels_csv
    merged, X, y, chromosomes, sample_ids, feature_cols, feature_names = load_features_and_merge_labels(
        features_path, labels_path, train_mode=True
    )
    mt = args.model

    if len(chromosomes) != X.shape[0]:
        print(f"ERROR: Chromosome info length ({len(chromosomes)}) doesn't match sample count ({X.shape[0]})")
        sys.exit(1)

    try:
        train_idx, test_idx = get_chromosome_train_test_split(chromosomes, seed=args.random_state)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    train_df = merged.iloc[train_idx].reset_index(drop=True)
    test_df = merged.iloc[test_idx].reset_index(drop=True)
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    ids_test = [sample_ids[i] for i in test_idx]
    chrom_train = [chromosomes[i] for i in train_idx]

    scaler = None
    if args.model in {"xgb", "l1", "enet"}:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        np.savetxt(os.path.join(output_dir, _out_name("scaler_mean.txt", mt)), scaler.mean_, fmt="%.6f")
        np.savetxt(os.path.join(output_dir, _out_name("scaler_scale.txt", mt)), scaler.scale_, fmt="%.6f")

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

    num_features = X_train.shape[1]
    if args.model == "xgb":
        dtrain_all = xgb.DMatrix(X_train, label=y_train)
        final_model = xgb.train(params, dtrain_all, num_boost_round=args.xgb_rounds)
        final_model.save_model(os.path.join(output_dir, _out_name("model.json", mt)))
        if scaler is not None:
            with open(os.path.join(output_dir, _out_name("scaler.pkl", mt)), "wb") as f:
                pickle.dump(scaler, f)
        final_importance = np.zeros(num_features, dtype=float)
        try:
            score_dict = final_model.get_score(importance_type="gain")
            for k, v in score_dict.items():
                if k.startswith("f"):
                    fi = int(k[1:])
                    if 0 <= fi < num_features:
                        final_importance[fi] = float(v)
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
            n_jobs=args.n_jobs,
            random_state=args.random_state,
        )
        final_model.fit(X_train, y_train)
        with open(os.path.join(output_dir, _out_name("model.pkl", mt)), "wb") as f:
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
            n_jobs=args.n_jobs,
        )
        final_model.fit(X_train, y_train)
        with open(os.path.join(output_dir, _out_name("model.pkl", mt)), "wb") as f:
            pickle.dump(final_model, f)
        if scaler is not None:
            with open(os.path.join(output_dir, _out_name("scaler.pkl", mt)), "wb") as f:
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
            n_jobs=args.n_jobs,
        )
        final_model.fit(X_train, y_train)
        with open(os.path.join(output_dir, _out_name("model.pkl", mt)), "wb") as f:
            pickle.dump(final_model, f)
        if scaler is not None:
            with open(os.path.join(output_dir, _out_name("scaler.pkl", mt)), "wb") as f:
                pickle.dump(scaler, f)
        try:
            final_importance = np.abs(final_model.coef_).ravel().astype(float)
        except Exception:
            final_importance = np.zeros(num_features, dtype=float)
        test_proba = final_model.predict_proba(X_test)[:, 1]

    pd.DataFrame({"feature": feature_names, "importance": final_importance}).to_csv(
        os.path.join(output_dir, "feature_importances.csv"), index=False
    )

    test_pred = (test_proba >= 0.5).astype(int)
    test_summary = compute_binary_metrics(y_test, test_proba)
    test_summary["train_size"] = int(len(train_idx))
    test_summary["test_size"] = int(len(test_idx))
    test_summary["train_chromosomes"] = int(train_df["chr"].astype(str).nunique())
    test_summary["test_chromosomes"] = int(test_df["chr"].astype(str).nunique())
    with open(os.path.join(output_dir, "test_summary.txt"), "w") as f:
        for k, v in test_summary.items():
            f.write(f"{k}\t{v}\n")

    pd.DataFrame(
        {"id": ids_test, "y_true": y_test.astype(int), "y_prob": test_proba, "y_pred": test_pred}
    ).to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)

    model_meta = {
        "model_type": args.model,
        "feature_cols": feature_cols,
        "num_features": int(X_train.shape[1]),
        "id_col": "name",
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
    }
    with open(os.path.join(output_dir, _out_name("model_meta.json", mt)), "w") as f:
        json.dump(model_meta, f, indent=2)
    print(f"Serialized model and {_out_name('model_meta.json', mt)} written under {output_dir}")
    print(f"Completed DeepSEA xQTL training. Outputs at: {output_dir}")
    
    


if __name__ == "__main__":
    main()
