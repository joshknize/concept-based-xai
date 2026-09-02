# A Feasibility Study of Concept-Based XAI Classification Methods on Chart Images

Josh Knize, Kenny Davila, Min Song

Supplementary code and notes for our paper evaluating two concept-based
explainability methods — **Concept Relevance Propagation** (CRP) and
**Label-free Concept Bottleneck Models** (LF-CBM) — on chart image
classification.

Chart images are a hard case for these methods on purpose: few non-synthetic
datasets, high visual similarity between classes, heavy class imbalance, and
enormous heterogeneity within a class. We adapted both frameworks to the
CHART-Info 2024 dataset and an OCC backbone, and found neither usable out of
the box — a wide interpretability gap, entangled concepts, poor concept
generation, and poor concept prediction.

This repository is a supplement, not a reproduction. It holds the pieces that
are useful to read on their own: the synthetic evaluation dataset generator,
the concept-prediction scoring code that the LF-CBM framework doesn't provide,
and notes on the adaptations each framework needed.

## Contents

| Path | What's there |
| --- | --- |
| [`synthetic-charts/`](synthetic-charts) | Seeded chart generator that emits per-image concept ground truth |
| [`lf-cbm-patches/`](lf-cbm-patches) | Concept-prediction scoring, plus notes on adapting LF-CBM to a ViT backbone |
| [`examples/`](examples) | Sample generated charts and the `labels.csv` they produced |
| [`figures/`](figures) | Figures from the paper |

## The main contribution here

LF-CBM labels each bottleneck neuron with a natural-language concept, but
neither the framework nor the original paper checks whether the neuron labeled
`a large circle` actually fires on images containing a large circle. Evaluation
is classification accuracy plus a crowdsourced study — and a model can score
well on both while its concepts mean nothing.

We built a ground truth to test that directly. `generate_charts.py` renders
charts with randomized styling and derives concept labels from the rendering
decisions themselves, so `colorful points` is true for exactly those scatter
charts that were drawn with a colorful palette. `concept_eval.py` then scores
the bottleneck against those labels, treating each (image, concept) pair as an
independent binary problem.

Our best classifier reached a **95.4% macro-F1** on chart classification — and
this is how its concepts scored:

| Concept activation threshold | Recall % | Precision % | F1 % | Accuracy % |
| --- | --- | --- | --- | --- |
| 0.0 | 89.6 | 24.5 | 43.3 | 48.5 |
| 0.5 | 84.7 | 28.1 | 46.0 | 57.3 |
| 1.0 | 73.0 | 31.4 | 45.9 | 66.2 |
| 1.5 | 52.8 | 34.1 | 40.4 | 74.7 |
| 2.0 | 28.9 | 34.5 | 29.1 | 79.6 |
| 2.5 | 15.4 | 36.0 | 18.5 | 82.6 |
| 3.0 | 5.3 | 26.9 | 8.6 | 82.6 |

Micro-averaged over all binary concept predictions. The accuracy column is a
trap: the no-information rate is **82.2%**, so the highest accuracy rows are
the ones where the model has all but stopped predicting concepts at all. F1
never clears 46%.

A state-of-the-art classifier, then, whose explanations don't track the image.
That's consistent with the information-leakage results in the CBM literature —
the bottleneck is routing information the concept labels don't describe. It
also means concept-model classification accuracy is not evidence that the
explanations are sound, which is how it's often reported.

## Concept Relevance Propagation

CRP produces relevance-ranked channels per class and visualizes each with its
most representative training samples. Below is the second most relevant channel
in `layer4.2.conv3` for predicting the `venn` class — 4.54% of the relevance
flowing into that prediction.

![CRP feature visualization for a venn chart concept](figures/crp-venn-concept.png)

Three problems this surfaces, detailed in the paper:

- **Layer selection.** Concepts live at different depths — a surface chart's
  3-D axes are a shallow feature while its rugged geometry is a deep one — and
  the framework gives no principled way to choose.
- **Entanglement.** The channel above holds at least two concepts (a circular
  edge, small text labels). Propagating backward to decompose it produced
  *more* entangled channels rather than cleaner primitives.
- **Dispersion.** That channel is 2nd most relevant *globally* but only 3rd in
  its own most-representative sample, at 1.9% — relevance is spread thin across
  2048 neurons, so global concept rankings say little about any single decision.

Chart data widens the interpretability gap further. The most representative
samples for a top horizontal-interval concept share a horizontal orientation,
aligned lines, points with whiskers, and a black-and-white palette — naming the
concept means picking one of those, and nothing in the output decides which.

![Representative samples for a horizontal interval chart concept](figures/crp-horizontal-interval.png)

CRP also has no contrastive interface: it shows what supported the predicted
class, never why the alternatives lost.

## LF-CBM

LF-CBM's explanation outputs are genuinely good — global concept-to-class flows
and per-image contributions:

![Sankey diagram of concept weights flowing into class predictions](figures/cbm-sankey.png)

The concepts flowing through them are the problem. GPT-generated concept sets
came back with heavy semantic overlap: tightening the cosine-similarity filter
from 0.9 to 0.8 cut 77 concepts to 24, but 7 of the 24 still referred to a
legend or key. Fine for prediction, bad for explanation — a local explanation
padded with four near-identical `axis` concepts explains less, not more.

![Local explanation for a single chart image](figures/cbm-local-explanation.png)

## Methods and data evaluated

- **Concept Relevance Propagation** — Achtibat et al., *Nature Machine
  Intelligence* 5(9), 2023. Built on Zennit.
- **Label-free Concept Bottleneck Models** — Oikarinen, Das, Nguyen & Weng,
  ICLR 2023.
- **CHART-Info 2024** — Davila et al., ICPR 2024.
- **OCC (chart classification backbone)** — Knize & Davila, *Optimizing Chart
  Image Classification: A Study of Data Augmentation and Training Strategies*,
  ICDAR 2025.

## Citation

```bibtex
@inproceedings{knize2026feasibility,
  title     = {A Feasibility Study of Concept-Based {XAI} Classification Methods on Chart Images},
  author    = {Knize, Josh and Davila, Kenny and Song, Min},
  booktitle = {Document Analysis Systems (DAS)},
  year      = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE). Code excerpts adapted from the Label-free-CBM
repository are marked as such in [`lf-cbm-patches/`](lf-cbm-patches); refer to
that project for its own terms.
