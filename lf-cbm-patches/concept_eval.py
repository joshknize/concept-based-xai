"""Score LF-CBM concept predictions against a concept ground truth.

The LF-CBM framework reports classification accuracy and evaluates its concept
layer qualitatively (plus a crowdsourced study). It has no way to ask the more
basic question: when the bottleneck neuron labeled "a large circle" fires, is
there actually a large circle in the image?

This module answers that. Given the concept activations a trained CBM produced
for a set of images, and a ground-truth CSV of the kind `generate_charts.py`
emits, it treats every (image, concept) pair as an independent binary
classification and reports precision / recall / F1 / accuracy -- both per
concept and micro-averaged -- across a sweep of activation thresholds.

The threshold sweep matters because there is no principled cutoff for "this
concept is present": the bottleneck emits an unbounded pre-activation value.
Sweeping it is what exposes the trade-off reported in the paper (Table 3),
where accuracy climbs to 82.6% only because the no-information rate is 82.2%
and the model has stopped predicting concepts almost entirely.

This file is extracted from the working fork and rewritten to stand alone --
it takes plain arrays instead of the fork's config objects, so it can be
pointed at any CBM's concept activations.

Usage:
    from concept_eval import evaluate_concepts, sweep_thresholds

    # acts: [n_images, n_concepts] pre-activation values from the bottleneck
    # concept_names: list of concept strings, in bottleneck column order
    # filenames: image basenames, in row order of `acts`
    per_concept, summary = evaluate_concepts(
        acts, concept_names, filenames, "labels.csv", threshold=1.0
    )
    print(sweep_thresholds(acts, concept_names, filenames, "labels.csv"))
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


DEFAULT_THRESHOLDS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)


def align_to_ground_truth(acts, concept_names, filenames, gt_csv):
    """Line up model concept activations with a ground-truth CSV.

    Two independent alignments are needed and both are easy to get wrong:

    1. Rows. `filenames` must be in the same order as the rows of `acts`. For
       a torchvision ImageFolder that means reading `dataset.samples` and
       iterating the DataLoader with shuffle=False -- a shuffled loader
       silently scrambles the pairing and produces plausible-looking garbage.
    2. Columns. The model's concept set and the ground-truth concept set
       overlap only partially, so we score the intersection and report how
       much of each set that covered.

    Returns (gt_matrix, pred_acts, shared_concepts, coverage), where
    `gt_matrix` is [n_matched, n_shared] of 0/1 and `pred_acts` is the
    corresponding slice of raw activations (not yet thresholded).
    """
    gt = pd.read_csv(gt_csv).set_index("filename")
    gt_cols = [c for c in gt.columns if c != "chart_type"]

    shared = [c for c in concept_names if c in gt_cols]
    if not shared:
        raise ValueError(
            "No concept names shared between the model and the ground truth.\n"
            f"  model (first 5): {concept_names[:5]}\n"
            f"  ground truth (first 5): {gt_cols[:5]}"
        )
    cols = [concept_names.index(c) for c in shared]

    gt_rows, act_rows = [], []
    for i, fname in enumerate(filenames):
        if fname in gt.index:
            gt_rows.append(gt.loc[fname, shared].values.astype(int))
            act_rows.append(np.asarray(acts[i])[cols])

    if not gt_rows:
        raise ValueError("No image filenames matched any row in the ground truth CSV.")

    coverage = {
        "images_matched": len(gt_rows),
        "images_total": len(filenames),
        "concepts_shared": len(shared),
        "concepts_in_model": len(concept_names),
        "concepts_in_gt": len(gt_cols),
    }
    return np.array(gt_rows), np.array(act_rows), shared, coverage


def evaluate_concepts(acts, concept_names, filenames, gt_csv, threshold=1.0):
    """Score concept predictions at one activation threshold.

    Returns (per_concept_df, summary_dict). `summary_dict` holds the
    micro-averaged scores -- every (image, concept) pair pooled into a single
    binary problem -- which is what the paper reports, along with the
    no-information rate for context.
    """
    gt_matrix, act_matrix, shared, coverage = align_to_ground_truth(
        acts, concept_names, filenames, gt_csv
    )
    pred_matrix = (act_matrix > threshold).astype(int)

    rows = []
    for j, concept in enumerate(shared):
        p, r, f, _ = precision_recall_fscore_support(
            gt_matrix[:, j], pred_matrix[:, j], average="binary", zero_division=0
        )
        rows.append({
            "concept": concept,
            "precision": p,
            "recall": r,
            "f1": f,
            "support": int(gt_matrix[:, j].sum()),
        })
    per_concept = pd.DataFrame(rows)

    p, r, f, _ = precision_recall_fscore_support(
        gt_matrix.ravel(), pred_matrix.ravel(), average="binary", zero_division=0
    )
    # Accuracy alone flatters a model on a sparse label matrix, so carry the
    # no-information rate (always predicting the majority class) alongside it.
    accuracy = float((gt_matrix == pred_matrix).mean())
    no_info_rate = float(max(gt_matrix.mean(), 1 - gt_matrix.mean()))

    summary = {
        "threshold": threshold,
        "recall": r,
        "precision": p,
        "f1": f,
        "accuracy": accuracy,
        "no_info_rate": no_info_rate,
        **coverage,
    }
    return per_concept, summary


def sweep_thresholds(acts, concept_names, filenames, gt_csv,
                     thresholds=DEFAULT_THRESHOLDS):
    """Run evaluate_concepts across thresholds; returns one row per threshold."""
    return pd.DataFrame([
        evaluate_concepts(acts, concept_names, filenames, gt_csv, t)[1]
        for t in thresholds
    ])[["threshold", "recall", "precision", "f1", "accuracy", "no_info_rate"]]
