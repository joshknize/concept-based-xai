# Concept sets

Every concept set we generated for the LF-CBM experiments, plus the filter
configuration that produced each one. These are the vocabularies the concept
bottleneck layer is built against, so they determine what the model's
explanations are able to say.

Two families are here: LLM-generated sets, following the LF-CBM recipe, and
CRP-informed sets, which we wrote by hand from the concepts we could actually
read out of CRP feature visualizations.

## Layout

| Path | Contents |
| --- | --- |
| `config/` | Filter settings, one JSON per concept set |
| `gpt3/init/` | Raw GPT-3.5 Turbo Instruct output, keyed by class |
| `gpt3/filtered/` | Final concept sets after filtering |
| `crp-informed/` | Hand-authored concept sets derived from CRP visualizations |
| `chartinfo_classes.txt` | The 15 CHART-Info 2024 classes the prompts were run over |

## How a set is built

LF-CBM prompts an LLM once per class, pools the results into one vocabulary,
then filters it down. The three prompts, each completed with a class name from
`chartinfo_classes.txt`:

| Key | Prompt |
| --- | --- |
| `important` | List the most important features for recognizing something as a {class} |
| `around` | List the things most commonly seen around a {class} |
| `superclass` | Give superclasses for the word {class} |

We appended "chart" to each class name, so the prompts read "…as a *surface
chart*". Output lands in `gpt3/init/` as `{class: [concepts]}`; the union is
then reduced by four filters:

- drop concepts longer than `max_len` characters
- drop concepts within `class_sim_cutoff` cosine similarity of any class name
- drop concepts within `other_sim_cutoff` cosine similarity of another concept
- (during training) drop concepts not activating on training data, and those
  the bottleneck cannot learn to project

The `other` key names an extra initial concept file to merge in, which is how
the CRP-informed concepts get mixed with the LLM ones.

## The sets

| Set | Sources | `max_len` | `class_sim` | `other_sim` | Concepts |
| --- | --- | --- | --- | --- | --- |
| `gpt3_v00_0` | important + around | 50 | 0.85 | 0.9 | 140 |
| `gpt3_v01_0` | important | 50 | 0.85 | 0.9 | 77 |
| `gpt3_v02_0` | around | 50 | 0.85 | 0.9 | 72 |
| `gpt3_v03_0` | important + around | 30 | 0.85 | 0.9 | 92 |
| `gpt3_v04_0` | important + around | 30 | 0.85 | 0.8 | 42 |
| `gpt3_v05_0` | `gpt3_v00_0`, manually filtered | — | — | — | 70 |
| `gpt3-crp_v00_0` | important + around + CRP-informed | 50 | 0.85 | 0.9 | 188 |
| `crp_v00_0` | CRP-informed only | 50 | 0.85 | 0.9 | 52 |

`class_sim` and `other_sim` are the LF-CBM defaults; `max_len` was raised from
30 to 50 for most runs because chart concepts are wordier than the object
concepts the framework was tuned on.

## What the filters didn't fix

`gpt3_v01_0` is the 77-concept set discussed in the paper. Thirteen of its 77
concepts mention a legend or a key. Tightening `other_sim_cutoff` from 0.9 to
0.8 removes most of the vocabulary — `gpt3_v04_0` keeps 42 of 92 — but the
redundancy survives the cut proportionally rather than being resolved by it.
Cosine similarity in the sentence-embedding space does not separate
"a legend or key explaining the markings" from "a legend or key to explain the
color coding" nearly as well as a reader would.

The `around` prompt is the clearer failure. `gpt3_v02_0` and the `around`
half of `gpt3_v00_0` contribute concepts like `a business suit`, `a cup of
coffee`, `a desk`, and `a magnifying glass` — things plausibly near a person
reading a chart, and never features of the chart image itself. They survive
filtering because they are short, dissimilar to class names, and dissimilar to
each other, which is everything the filters check.

## CRP-informed sets

`crp-informed/` holds the concepts we wrote by reading CRP feature
visualizations directly, as an attempt at a vocabulary grounded in what the
network actually represents.

| File | Concepts | What it is |
| --- | --- | --- |
| `crp-informed-concepts.json` | 111 | Per-class, before filtering — the input to `crp_v00_0` |
| `crp_v00_0.txt` | 52 | Filtered flat vocabulary |
| `crp_v01_0.txt` | 28 | Subset with programmatically labelable ground truth |

`crp_v01_0.txt` is the vocabulary the synthetic evaluation runs against; it is
copied to [`../synthetic-charts/concepts.txt`](../synthetic-charts/concepts.txt)
so the generator is self-contained. Its typos (`overlapping cirlces`) are
preserved deliberately — the concept strings are keys, and the trained
bottleneck columns are labeled with them.

These sets are more visually grounded than the LLM output, but writing them
required exactly the manual interpretation of feature visualizations that the
paper argues is the weak point of CRP. They are a diagnostic, not a solution.
