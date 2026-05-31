import pandas as pd
import numpy as np
import sys
import os

# -------------------------------
# 1. Read input files
# -------------------------------
df_importance = pd.read_csv(sys.argv[1], sep=",")  # feature_importance.txt
df_annotation = pd.read_csv(sys.argv[2], sep="\t")  # feature_to_tissue.txt

# Merge importance and annotation
df = df_importance.merge(df_annotation, on="feature", how="inner")

# -------------------------------
# 2. Define top 10% high-importance features
# -------------------------------
df["abs_importance"] = df["importance"].abs()
TOP_PCT = 0.10
num_top = int(len(df) * TOP_PCT)

df = df.sort_values("abs_importance", ascending=False)
df["is_high"] = 0
df.iloc[:num_top, df.columns.get_loc("is_high")] = 1

# -------------------------------
# 3. Function: observed enrichment
# -------------------------------
def calculate_observed_enrichment(df, label_col):
    high_features = df[df["is_high"] == 1]
    return high_features[label_col].value_counts(normalize=True)

# -------------------------------
# 4. Function: permutation enrichment (sample from non-top10% features)
# -------------------------------
def permutation_enrichment_non_top(df, label_col, n_permutations=10000, random_state=1):
    rng = np.random.default_rng(random_state)
    perm_results = []

    non_top_indices = df[df["is_high"] == 0].index
    num_high = df["is_high"].sum()

    for _ in range(n_permutations):
        # Randomly sample K features from non-top10%
        sampled_indices = rng.choice(non_top_indices, size=num_high, replace=False)
        df_perm = df.copy()
        df_perm["is_high"] = 0
        df_perm.loc[sampled_indices, "is_high"] = 1

        # Calculate enrichment fraction
        enrich = df_perm[df_perm["is_high"] == 1][label_col].value_counts(normalize=True)
        perm_results.append(enrich)

    perm_df = pd.concat(perm_results, axis=1).T.fillna(0)
    return perm_df

# -------------------------------
# 5. Function: empirical p-value and FDR
# -------------------------------
def compute_empirical_p_and_fdr(observed, perm_df):
    # Ensure perm_df has the same columns as observed, fill missing columns with 0
    perm_df = perm_df.reindex(columns=observed.index, fill_value=0)

    p_values = {}
    for label in observed.index:
        obs_value = observed[label]
        perm_values = perm_df[label]
        p_val = (1 + np.sum(perm_values >= obs_value)) / (1 + len(perm_values))
        p_values[label] = p_val

    results = pd.DataFrame({
        "Enrichment": observed,
        "Empirical_P": pd.Series(p_values)
    })
    # Simple FDR correction
    results["FDR"] = (results["Empirical_P"].rank(method="min") / len(results)) * results["Empirical_P"]
    results = results.sort_values("Empirical_P")
    return results

# -------------------------------
# 6. Run enrichment for multiple annotation columns
# -------------------------------
annotation_cols = ["assay","tissue"]
output_dir = sys.argv[3]  # directory to save results
os.makedirs(output_dir, exist_ok=True)

for col in annotation_cols:
    print(f"Processing enrichment for {col}...")
    observed = calculate_observed_enrichment(df, col)
    perm_df = permutation_enrichment_non_top(df, col, n_permutations=10000)
    results = compute_empirical_p_and_fdr(observed, perm_df)
    out_file = os.path.join(output_dir, f"feature_enrichment_{col}.tsv")
    results.to_csv(out_file, sep="\t")
    print(f"Saved enrichment results for {col} -> {out_file}")

print("All enrichment analyses completed.")
