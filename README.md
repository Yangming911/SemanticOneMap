# SemanticOneMap: OACP Semantic Obstacle Avoidance for Object Navigation

Built on [OneMap](https://github.com/KTH-RPL/OneMap). This fork adds **Online Adaptive Conformal Prediction (OACP)** for semantic obstacle detection and avoidance during zero-shot object navigation.

## Method Overview

The baseline OneMap planner navigates using only geometric (depth) obstacle maps and has no awareness of semantic obstacles — furniture, fixtures, etc. that the robot should not collide with. We add a GCLIP-based semantic obstacle map that identifies and inflates obstacle regions on the navigable map so the planner routes around them.

Three variants are implemented:

| Variant | Obstacle Detection Logic | Config |
|---------|------------------------|--------|
| **Baseline** | None (depth-only) | `eval_baseline.yaml` |
| **GCLIP Argmax** | `argmax(obs_sim) > max(bg_sim)` | `eval_argmax.yaml` |
| **OACP (ours)** | Conformal prediction set with ACI-calibrated margin | `eval_oacp.yaml` |

### Key Components

- **`mapping/clip_cp_obstacle_map.py`** — CLIP-based Conformal Prediction obstacle map. Maintains per-cell nonconformity scores, ACI threshold adaptation, and margin-based confidence sets (Path A). Falls back to argmax or threshold modes via config.

- **`eval/semantic_collision.py`** — Ground-truth semantic collision judge. Per-class obstacle dictionary with tuned inflation radii. Includes cross-floor phantom filter (`_MAX_BOTTOM_BELOW_FLOOR`) to prevent false positives from objects on different storeys.

- **`mapping/navigator.py`** — Integrates CLIP-CP obstacle mask into the planner's navigable map. Supports detection gate (activate only after target seen) and step guard (lookahead collision check).

### Obstacle Dictionary (dict4)

Per-class radii tuned via baseline trajectory pass-through analysis:

```python
_SEMANTIC_SAFETY_RADIUS_CELLS = {
    "shower":           3,
    "cabinet":          4,
    "chest of drawers": 4,
    "table":            5,
    "tv":               2,
}
```

## Setup

### Prerequisites

- CUDA GPU (tested on RTX 3090)
- Conda environment with Python 3.8
- Habitat-sim v0.2.4
- Matterport3D scene data

### Install

```bash
conda create -n onemap python=3.8
conda activate onemap
pip install torch torchvision torchaudio
pip install -r requirements.txt
CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" pip install git+https://github.com/facebookresearch/habitat-sim.git@v0.2.4
pip install --upgrade timm>=1.0.7
pip install ./planning_cpp/
git clone https://github.com/WongKinYiu/yolov7
```

### Weights

```bash
mkdir -p weights/
gdown 1D_RE4lvA-CiwrP75wsL8Iu1a6NrtrP9T -O weights/clip.pth
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt -O weights/yolov7-e6e.pt
wget https://github.com/ChaoningZhang/MobileSAM/raw/refs/heads/master/weights/mobile_sam.pt -O weights/mobile_sam.pt
```

### GCLIP Model

The GCLIP obstacle map uses OpenAI's CLIP ViT-B/16 via [open_clip](https://github.com/mlfoundations/open_clip). Weights are downloaded automatically on first run:

```bash
pip install open_clip_torch
```

No separate weight file needed — `open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')` fetches and caches the checkpoint automatically.

### Data

Place MP3D scene datasets under `datasets/scene_datasets/mp3d/` and ObjectNav episodes under `datasets/objectnav_mp3d_v1/`.

## Running Experiments

All experiments use `val_mini` (11 scenes, 33 episodes).

```bash
# Baseline (no semantic obstacles)
xvfb-run -a python -u eval_habitat.py -c config/mon/eval_baseline.yaml

# Argmax ablation (GCLIP argmax, no conformal prediction)
xvfb-run -a python -u eval_habitat.py -c config/mon/eval_argmax.yaml

# OACP (full method)
xvfb-run -a python -u eval_habitat.py -c config/mon/eval_oacp.yaml
```

Results are saved to `results/<config_name>/state/state_<ep>.txt` (1=SUCCESS, 7=COLLISION).

### Visualization

```bash
# Single episode video (e.g., ep31 where OACP succeeds and baseline collides)
xvfb-run -a python -u visualize_single_scene.py \
    --episode-id 31 --no-display --output outputs/ep31.mp4 \
    -c config/mon/eval_oacp.yaml
```

## Results (MP3D val_mini, 33 episodes)

| Method | Collisions | Collision Rate | Success | Success Rate |
|--------|-----------|---------------|---------|-------------|
| Baseline | 7 | 21.2% | 2 | 6.1% |
| GCLIP Argmax | 5 | 15.2% | 4 | 12.1% |
| **OACP (ours)** | **3** | **9.1%** | **4** | **12.1%** |

Collision reduction: baseline → argmax **−29%**, baseline → OACP **−57%**.

### Per-class breakdown

| Class | Baseline | Argmax | OACP |
|-------|----------|--------|------|
| table | 3 | 1 | 0 |
| shower | 3 | 2 | 2 |
| cabinet | 1 | 1 | 1 |
| chest of drawers | 0 | 1 | 0 |

## Project Structure

```
config/mon/                  # Experiment configs
eval/
  semantic_collision.py      # GT collision judge + phantom fix
  habitat_evaluator.py       # Main evaluation loop
mapping/
  clip_cp_obstacle_map.py    # CLIP-CP / OACP obstacle map
  navigator.py               # Map integration + planner interface
analysis/                    # Dict tuning & phantom verification scripts
visualize_single_scene.py    # Episode visualization with obstacle panels
```

## Acknowledgements

Built on [OneMap](https://github.com/KTH-RPL/OneMap) by Busch et al. (2024).
