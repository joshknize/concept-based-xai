import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


DEFAULT_THRESHOLDS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)


def align_to_ground_truth(acts, concept_names, filenames, gt_csv):

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
