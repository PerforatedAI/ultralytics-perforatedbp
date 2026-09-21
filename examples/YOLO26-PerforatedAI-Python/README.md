# YOLO26 with PerforatedAI Dendrites

This example trains [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26) with [PerforatedAI](https://www.perforatedai.com/) artificial dendrites on Cityscapes (set up as an 8-class object detection task). Each of the four YOLO26 sizes is fine-tuned twice from its COCO checkpoint, once plain and once with dendrites, so the two arms differ only in whether dendrites are on. Dendrite training is switched on with the `perforate=True` train argument. Every PerforatedAI setting is a normal train argument, listed under "PerforatedAI settings" in `ultralytics/cfg/default.yaml`, so it works from the CLI, from `YOLO(...).train(...)`, and from a `cfg=` YAML.

## Dataset

Cityscapes is a segmentation benchmark: 5,000 finely annotated street scenes at 2048x1024 with pixel masks for 30 classes. The `cityscapes.yaml` that ships with Ultralytics is the 19-class semantic set built from those masks. While there isn't an official detection split, our earlier dendrite experiments used the mmdetection conversion, which keeps the 8 instance classes (person, rider, car, truck, bus, train, motorcycle, bicycle), draws a tight box around each instance mask, and drops crowd regions the way COCO drops `iscrowd`. We rebuilt this in the YOLO layout so the new runs score against the old ones: 2,975 training images and the 500 official validation images, since the test labels are not public.

`prepare_cityscapes_det.py` reproduces the conversion without going through COCO JSON. It reads each `_gtFine_instanceIds.png`, takes instance ids at or above 24 as objects, decodes the class as `id // 1000`, drops ids under 1000 as crowd regions, boxes each mask, and writes YOLO labels in mmdetection's class order. The numbers provided below are comparable to other mmdetection-style Cityscapes detection results but not to the Cityscapes segmentation leaderboard.

## Results

Cityscapes 8-class detection, best mAP50-95 on the 500 validation images. Each row is one plain arm and one dendrite arm trained from the same config in this folder, differing only in `perforate`. Dendrite scores are reported at 2 dendrites, except YOLO26s where the run kept only 1 dendrite.

| Model   | Plain mAP50-95 | Dendrites mAP50-95 | Dendrites added | Plain params | Dendrite params | Param increase | Gain (points) | Gap to next size closed |
| ------- | -------------- | ------------------ | --------------- | ------------ | --------------- | -------------- | ------------- | ----------------------- |
| YOLO26n | 0.2620         | 0.2732             | 2               | 2,506,920    | 2,636,136       | +5.2%          | +1.1          | 20% of n to s           |
| YOLO26s | 0.3181         | 0.3292             | 1               | 9,954,056    | 10,212,296      | +2.6%          | +1.1          | 27% of s to m           |
| YOLO26m | 0.3597         | 0.3779             | 2               | 21,785,224   | 23,260,552      | +6.8%          | +1.8          | 56% of m to l           |
| YOLO26l | 0.3924         | 0.4102             | 2               | 26,188,680   | 27,664,008      | +5.6%          | +1.8          | N/A                     |

These results are from Perforated experiments focusing on maximizing performance gains and minimizing parameters added. The plain rows trace the normal size-versus-accuracy curve, where going from n to s costs 4x the parameters for 5.6 points. Dendrites add a few percent of parameters and move every size above that curve. YOLO26n with dendrites gets 20% of the way to YOLO26s accuracy at about a quarter of the YOLO26s parameter count. The lift holds at every size, so it is a shift of the whole Pareto front rather than a nano-only effect.

## Files

- `train_pai.py`: Minimal example that perforates every `Conv` in the model with every PerforatedAI key at its default.
- `yolo26n_cityscapes_pai.yaml`: The YOLO26n Cityscapes recipe. Dendrites on the first `Conv` of the `one2one` box branch at each pyramid level, AdamW with no weight decay, and a plateau learning rate schedule on mAP50-95. One file serves both arms.
- `cityscapes-det.yaml`: The 8-class Cityscapes detection set (not the same as the 19-class semantic cityscapes.yaml that Ultralytics has).
- `prepare_cityscapes_det.py`: Downloads Cityscapes and converts the instance masks to YOLO labels for `cityscapes-det.yaml`.

## Walkthrough

All commands run from the repository root.

### 1. Install

```bash
pip install -e .
```

`perforatedai` and `perforatedbp` are pinned in `pyproject.toml` and install with the package. `perforated_backpropagation=True`, the default, needs a `perforatedbp` license. Set it to `False` to train dendrites in open-source mode.

### 2. Build Cityscapes

Cityscapes needs a free account at [cityscapes-dataset.com](https://www.cityscapes-dataset.com/). The script downloads `leftImg8bit` and `gtFine`, then writes the 8-class detection labels.

```bash
python examples/YOLO26-PerforatedAI-Python/prepare_cityscapes_det.py --download \
    --cityscapes-root datasets/cityscapes-raw \
    --out-root datasets/cityscapes-det
```

`--out-root` must match the `path` key in `cityscapes-det.yaml`, resolved against the Ultralytics datasets directory.

### 3. Train the two arms

```bash
# YOLO26n (Dendrites)
yolo train cfg=examples/YOLO26-PerforatedAI-Python/yolo26n_cityscapes_pai.yaml name=yolo26n_cityscapes_pai perforate=True

# YOLO26n (Vanilla)
yolo train cfg=examples/YOLO26-PerforatedAI-Python/yolo26n_cityscapes_pai.yaml name=yolo26n_cityscapes_plain

# YOLO26s (Vanilla) (Next size for parameter comparison)
yolo train cfg=examples/YOLO26-PerforatedAI-Python/yolo26n_cityscapes_pai.yaml model=yolo26s.pt batch=32 name=yolo26s_cityscapes_plain
```

The vanilla run stops on `patience` and the dendrite run will stop when PerforatedAI reports that the last dendrite did not improve the score. It usually takes 75 - 150 epochs for the Vanilla runs to complete and 250 - 450 epochs for the dendrite runs to complete.

### 4. Read the results

- `runs/detect/<name>/results.csv` has the per-epoch mAP50-95. The dendrite arm's mAP is flat during dendrite phases because the neurons are frozen; the plateau schedule tracks correlation instead during those epochs.
- `runs/detect/<name>/weights/best.pt` is the best epoch, with dendrites folded into a deep copy of the EMA model.
- The PerforatedAI folder `<name>_pai/` holds the score graphs, `*_best_arch_scores.csv` with the best score per dendrite count, and `*param_counts.csv` with the parameter count at each switch. Those two CSVs are the source of the params and mAP columns in the results table above.
