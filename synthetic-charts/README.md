# Synthetic charts with concept ground truth

A seeded matplotlib generator that renders chart images across five classes
and labels each one against a concept vocabulary — producing the ground truth
that concept-bottleneck predictions can actually be scored against.

```bash
python generate_charts.py                  # uses config/chart_gen_config.json
python generate_charts.py -n 5             # small run, 5 charts per class
python generate_charts.py --config my.json
```

Requires `matplotlib` and `numpy`. Output lands under the per-class
`output_dir` paths in the config, with a single `labels.csv` alongside them.

## Why it exists

LF-CBM's bottleneck neurons are each labeled with a natural-language concept,
but nothing in the framework checks whether a neuron labeled `a large circle`
fires on images that contain a large circle. The original evaluation is a
crowdsourced study; the paper's argument is that classification accuracy and
user ratings both leave that question open.

Real chart datasets don't carry concept annotations, and hand-labeling
thousands of images against dozens of concepts isn't practical. Generating the
images instead means the concept labels fall out of the rendering decisions for
free, and they're exact rather than estimated.

## How labels are assigned

Each chart maker returns the styling metadata it actually used — colormap,
point count, series count, whether a legend was drawn, whether markers were
placed. `CONCEPT_RULES` maps that metadata to concepts, per chart class:

```python
"colorful points":  lambda m: m["color_mode"] in ("by_group", "random_per_point",
                                                  "by_value"),
"densely packed scatter points": lambda m: m.get("n_points", 0) >= 150,
"dashed line":      lambda m: m["line_style"] in ("--", "-.", ":"),
```

The distinction that matters: labels are conditional on what was drawn, not
assumed from the class. Every pie chart gets `a large circle`, but only the
scatter charts whose random style happened to pick a vibrant palette get
`colorful points`. Concepts a class can never exhibit default to 0.

This keeps the concept labels genuinely varied within a class, so a bottleneck
can't score well by memorizing the class and emitting its concept template.

## Files

| Path | What it is |
| --- | --- |
| `generate_charts.py` | Generator, concept rules, and CSV writer |
| `concepts.txt` | Concept vocabulary, one per line (CRP-informed set from the paper) |
| `config/chart_gen_config.json` | Per-class counts, styling ranges, output paths, seed |

Sample output — one chart per class plus the `labels.csv` they produced — is in
[`../examples/`](../examples).

## Notes

- Concept strings in `CONCEPT_RULES` must match `concepts.txt` byte for byte,
  including the typos carried over from the original concept set. To use a
  different vocabulary, replace `concepts.txt` and update the rules together.
- Generation is seeded from the config, so a given config reproduces the same
  charts and the same labels.
- `concepts.txt` holds 28 concepts; only the subset a class can plausibly
  exhibit is ever labeled positive, so a run's usable ground truth is narrower
  than the full vocabulary. The paper's evaluation used 16 concepts across 500
  charts (100 per class); the committed config uses different per-class counts.
