#!/usr/bin/env python3
import os
import sys
import argparse
import gzip
from typing import Iterator, Dict, Any

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import pyfaidx

# Minimal one-hot encoder for ACGT (N->0s)
def one_hot_encode(sequence: str) -> np.ndarray:
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    arr = np.zeros((len(sequence), 4), dtype=np.float32)
    for i, b in enumerate(sequence.upper()):
        j = mapping.get(b)
        if j is not None:
            arr[i, j] = 1.0
    return arr


SEQUENCE_LENGTH = 393_216


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score variants in a VCF using Enformer and export PC20 or full 5313-track normalized scores.")
    p.add_argument("--vcf", required=True, help="Path to VCF (.vcf or .vcf.gz)")
    p.add_argument("--fasta", required=True, help="Reference genome FASTA matching VCF build (e.g., hg38.fa)")
    p.add_argument("--model", default="https://tfhub.dev/deepmind/enformer/1", help="TFHub URL or local SavedModel path for Enformer")
    p.add_argument("--organism", choices=["human", "mouse"], default="human", help="Output head to use")
    p.add_argument("--mode", choices=["pc20", "full"], default="pc20", help="pc20: output top 20 PCs; full: output all 5313 normalized scores")
    p.add_argument("--targets", default="https://raw.githubusercontent.com/calico/basenji/0.5/manuscripts/cross2020/targets_human.txt", help="targets_{organism}.txt URL or local path (only used for full mode header naming)")
    p.add_argument("--max_variants", type=int, default=None, help="Optional cap on the number of variants to score (for testing)")
    p.add_argument("--out", required=True, help="Output CSV path")
    p.add_argument("--device", default=None, help="Force device, e.g. /CPU:0 or /GPU:0 (default: auto)")
    p.add_argument("--threads", type=int, default=0, help="Set TF intra/inter op threads for CPU (0=leave default)")
    return p.parse_args()


def iter_vcf(path: str) -> Iterator[Dict[str, Any]]:
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt') as f:
        for line in f:
            if not line or line.startswith('#'):
                continue
            parts = line.rstrip().split('\t')
            if len(parts) < 5:
                continue
            chrom, pos, vid, ref, alts = parts[:5]
            for alt in alts.split(','):
                yield {
                    'chrom': chrom,
                    'pos': int(pos),
                    'id': vid,
                    'ref': ref,
                    'alt': alt
                }


def load_targets(targets_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(targets_path, sep='\t')
    except Exception:
        return pd.DataFrame()


def extract_centered_sequences(fasta: pyfaidx.Fasta,
                               chrom: str,
                               pos_1based: int,
                               ref: str,
                               alt: str) -> Dict[str, np.ndarray]:
    """
    Extract a fixed-length (SEQUENCE_LENGTH) window centered at the variant.
    If the window runs off the chromosome edges, pad with 'N' on left/right
    so the final sequence length is exactly SEQUENCE_LENGTH.
    Perform a simple allele edit to create the alt sequence:
      - If len(ref) == len(alt): do an in-place replacement of that span
      - Else (indel/unequal length): replace only the center base
    Return one-hot encoded arrays of shape (SEQUENCE_LENGTH, 4) for both ref and alt.
    """
    half = SEQUENCE_LENGTH // 2
    center0 = pos_1based - 1           # 0-based center coordinate
    chrom_len = len(fasta[chrom])      # chromosome length

    # Desired 0-based [start, end) window
    start0 = center0 - half
    end0   = center0 + half            # half-open end

    # Fetchable bounds (pyfaidx is 1-based inclusive)
    fetch_start_1 = max(start0, 0) + 1
    fetch_end_1   = min(end0, chrom_len)

    # Fetch the available subsequence
    seq_core = fasta.get_seq(chrom, fetch_start_1, fetch_end_1).seq.upper()

    # Pad with 'N' to guarantee fixed length
    left_pad  = max(0, -start0)                  # how many bases ran off the left
    right_pad = max(0, end0 - chrom_len)         # how many bases ran off the right
    seq_ref = ('N' * left_pad) + seq_core + ('N' * right_pad)

    # Safety check
    if len(seq_ref) != SEQUENCE_LENGTH:
        # In case of unexpected off-by-one, force the exact length
        if len(seq_ref) > SEQUENCE_LENGTH:
            seq_ref = seq_ref[:SEQUENCE_LENGTH]
        else:
            seq_ref = seq_ref + ('N' * (SEQUENCE_LENGTH - len(seq_ref)))

    # Build alt sequence by editing around the center
    seq_list = list(seq_ref)
    if len(ref) == len(alt) and len(ref) > 0:
        # Replace the span [center, center+len(ref))
        for k in range(len(ref)):
            seq_list[half + k] = alt[k]
    else:
        # Fallback: replace only the center base
        if len(alt) > 0:
            seq_list[half] = alt[0]

    seq_alt = ''.join(seq_list)

    # One-hot encode (A,C,G,T; N -> zeros)
    return {
        'ref': one_hot_encode(seq_ref),
        'alt': one_hot_encode(seq_alt),
    }

def main():
    args = parse_args()

    if args.threads and args.threads > 0:
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
        tf.config.threading.set_inter_op_parallelism_threads(args.threads)

    device = args.device if args.device else ("/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0")

    fasta = pyfaidx.Fasta(args.fasta)
    df_targets = load_targets(args.targets) if args.mode == 'full' else pd.DataFrame()

    with tf.device(device):
        model = hub.load(args.model)

        rows = []
        for i, v in enumerate(iter_vcf(args.vcf)):
            if args.max_variants is not None and i >= args.max_variants:
                break
            seqs = extract_centered_sequences(fasta, v['chrom'], v['pos'], v['ref'], v['alt'])

            ref_pred = model.model.predict_on_batch(seqs['ref'][np.newaxis])[args.organism][0]
            alt_pred = model.model.predict_on_batch(seqs['alt'][np.newaxis])[args.organism][0]
            raw = (tf.reduce_mean(alt_pred, axis=0) - tf.reduce_mean(ref_pred, axis=0)).numpy()[np.newaxis, :] # [1, num_tracks]

            if args.mode == 'pc20':
                # Without an external transform, emit the first 20 raw track deltas as PC1..PC20.
                scores = raw[0][:20]
                row = {**v}
                for j in range(20):
                    row[f'PC{j+1}'] = float(scores[j])
                rows.append(row)
            else:
                # Emit all raw track deltas, using target descriptions when available.
                scores = raw[0]
                row = {**v}
                for idx, val in enumerate(scores):
                    if not df_targets.empty and idx < len(df_targets):
                        name = str(df_targets.description.iloc[idx])
                    else:
                        name = f"track_{idx}"
                    row[name] = float(val)
                rows.append(row)
            if (i + 1) % 1000 == 0:
                print(f"Scored {i+1} variants", file=sys.stderr)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} variants to {args.out}")


if __name__ == "__main__":
    main()