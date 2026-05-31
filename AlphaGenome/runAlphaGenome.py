import os
import sys
import numpy as np
import pandas as pd
from alphagenome.data import genome
from alphagenome_research.model import dna_model
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# -------------------------------
# Arguments
# -------------------------------
vcf_file = sys.argv[1]
output_dir = sys.argv[2]
os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# Load model
# -------------------------------
model = dna_model.create('/home/shchen/Aim1/AlphaGenome/alphagenome-jax-all_folds-v1/')
metadata = model._metadata[dna_model.Organism.HOMO_SAPIENS]
all_outputs = list(dna_model.OutputType)

# -------------------------------
# Read VCF
# -------------------------------
vcf_df = pd.read_csv(
    vcf_file, 
    sep=r"\s+", 
    header=None, 
    names=['chrom', 'pos', 'id', 'ref', 'alt']
)

# -------------------------------
# Parameters
# -------------------------------
half_window = 65536 
desired_length = 2 * half_window
feature_list = [
    "ATAC", "CAGE", "DNASE", "RNA_SEQ",
    "CHIP_HISTONE", "CHIP_TF",
    "SPLICE_SITES", "SPLICE_SITE_USAGE", "SPLICE_JUNCTIONS",
    "CONTACT_MAPS", "PROCAP"
]
batch_size = 5

# -------------------------------
# Aggregate function
# -------------------------------
def aggregate_delta(delta_array):
    if delta_array.ndim == 2:
        return delta_array.mean(axis=0),  np.abs(delta_array).max(axis=0)
    elif delta_array.ndim == 3:
        return delta_array.mean(axis=(0,1)),  np.abs(delta_array).max(axis=(0,1))
    else:
        raise ValueError(f"Unsupported delta_array shape {delta_array.shape}")

# -------------------------------
# Batch prediction loop
# -------------------------------
feature_dfs = {feat: [] for feat in feature_list}
n_snps = len(vcf_df)
for start_idx in range(0, n_snps, batch_size):
    end_idx = min(start_idx + batch_size, n_snps)
    batch_rows = vcf_df.iloc[start_idx:end_idx]

    for row in batch_rows.itertuples(index=False):
        # Define interval and variant
        start = max(1, row.pos - half_window)
        end = start + desired_length
        interval = genome.Interval(chromosome=row.chrom, start=start, end=end)
        variant = genome.Variant(
            chromosome=row.chrom,
            position=row.pos,
            reference_bases=row.ref,
            alternate_bases=row.alt
        )

        # Predict outputs for all features
        outputs = model.predict_variant(
            interval=interval,
            variant=variant,
            ontology_terms=None,
            requested_outputs=all_outputs,
        )

        # Process each feature
        for feature in feature_list:
            ref_vals = getattr(outputs.reference, feature.lower()).values
            alt_vals = getattr(outputs.alternate, feature.lower()).values
            delta_all = alt_vals - ref_vals
            if delta_all.size == 0: 
                continue 
            foldchange_all = (alt_vals+1e-6)/(ref_vals+1e-6)
            delta_mean, delta_max = aggregate_delta(delta_all)
            foldchange_mean,foldchange_max = aggregate_delta(foldchange_all)
            track_names = getattr(outputs.reference, feature.lower()).metadata["name"].values
            df = pd.DataFrame({
                "id": [row.id]*len(track_names),
                "feature_name": track_names,
                "delta_mean": delta_mean,
                "delta_max": delta_max,
                "foldchange_mean": foldchange_mean,
                "foldchange_max": foldchange_max
            })
            feature_dfs[feature].append(df)
    if (start_idx + 5) % 1000 == 0:
        print(f"Processed {start_idx + 5}/{n_snps} SNPs")

# -------------------------------
# Write CSVs
# -------------------------------
for feature, dfs in feature_dfs.items():
    output_file = os.path.join(output_dir, f"{feature}_131kb.csv")
    pd.concat(dfs, ignore_index=True).to_csv(output_file, index=False)
