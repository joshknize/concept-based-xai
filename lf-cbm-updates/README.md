# LF-CBM adaptations

Notes on the changes needed to run Label-free Concept Bottleneck Models
(Oikarinen et al., ICLR 2023) against a DINOv2 backbone and a chart-image
dataset, plus the evaluation code that isn't in the upstream framework.

These are excerpts, not a fork. They're here so the paper's claims are
inspectable without needing the full working repo.

## 1. Transformer backbones produce 3-D features

Upstream assumes a CNN backbone and strips the final FC layer and flattens
whatever comes out, which for a ResNet is `[B, C, 1, 1] -> [B, C]`. A ViT's
`forward_features` returns `[B, seq_len, dim]` instead. The fix is to branch 
on rank and take the CLS token, in `CBM_model.forward` and `standard_model.forward` 
in `cbm.py`:

```python
def forward(self, x):
    x = self.backbone(x)

    # Handle ViT 3D output (extract CLS token)
    if len(x.shape) == 3:
        x = x[:, 0, :]           # [B, seq_len, dim] -> [B, dim]
    else:
        x = torch.flatten(x, 1)  # ResNet [B, C, 1, 1] -> [B, C]

    x = self.proj_layer(x)
    proj_c = (x - self.proj_mean) / self.proj_std
    x = self.final(proj_c)
    return x, proj_c
```

The backbone itself also needs a separate case, since the OCC DINOv2 model
must be called through `forward_features` rather than by chopping off
`children()[-1]`:

```python
elif "occ" in backbone_name:
    self.backbone_model = model.model
    self.backbone = self.backbone_model.forward_features
```

The same 3-D assumption is baked into the activation-saving path used to build
the concept layer, so that has to be handled in `utils.get_activation`, too.

## 2. Scoring concepts against a ground truth

[`concept_eval.py`](concept_eval.py) is the piece with no upstream equivalent.
It treats each (image, concept) pair as an independent binary classification
and scores the bottleneck against the labels emitted by
[`../synthetic-charts/generate_charts.py`](../synthetic-charts/generate_charts.py),
sweeping the activation threshold because there's no principled cutoff for
"this concept is present."

```python
from concept_eval import sweep_thresholds

# acts: [n_images, n_concepts] bottleneck pre-activations
# concept_names: bottleneck column order; filenames: row order of acts
print(sweep_thresholds(acts, concept_names, filenames, "labels.csv"))
```

It scores the intersection of the model's concept set and the ground-truth
concept set, and reports how much of each that covered. It also carries the
no-information rate next to accuracy, because on a sparse label matrix
accuracy is close to meaningless on its own.

## Attribution

Snippets in sections 1 and 2 are modifications of code from the Label-free-CBM
repository (Oikarinen et al., ICLR 2023); refer to that repository for its
license terms. `concept_eval.py` is original work.
