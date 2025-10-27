This repository provides tools for converting **drone-view traffic datasets** into formats suitable for **autonomous driving simulation** and **motion prediction**.
It supports multiple datasets and output formats, handling trajectory data, map features, and agent metadata seamlessly.

---

## ✨ Features

* **Multi-Dataset Support**: inD, INTERACTION, and SinD

* **Multi-Format Output**:

  * [ScenarioNet](https://github.com/metadriverse/scenarionet)
  * [MetaDrive](https://github.com/metadriverse/metadrive)
  * [VBD](https://github.com/SafeRoboticsLab/VBD)
  * [Scenario Dreamer](https://github.com/princeton-computational-imaging/scenario-dreamer)

* **Upcoming Support**:

  * [ScenarioMax](https://github.com/valeoai/V-Max/tree/main/vmax)
  * [GPUDrive](https://github.com/Emerge-Lab/gpudrive)
  * [trajdata](https://github.com/NVlabs/trajdata)

* **Ego-Centric Alignment**: Automatic ego vehicle selection and coordinate transformation

* **Segmentation**: Split long recordings into manageable scenario clips

---

## 📂 Project Structure

```
drone-tool/
├── scenarionet-converter/                  # Convert drone datasets to ScenarioNet format
│   ├── inD_scene.py
│   ├── interaction_scene.py
│   └── sind_scene.py
├── scenarionet-VBD-converter/              # Convert ScenarioNet → VBD
│   └── convert_scenarionet_to_vbd.py
└── scenarionet-scenariodreamer-converter/  # Convert ScenarioNet → Scenario Dreamer
    └── scenarionet_to_scenariodreamer_waymo.py
```

---

## ⚙️ Installation

### Core Dependencies

```bash
pip install numpy pandas scipy shapely lxml utm tqdm matplotlib omegaconf
```

### ScenarioNet / MetaDrive

```bash
pip install metadrive-simulator scenarionet
```

---

## 🚀 Usage

### 1. Convert inD → ScenarioNet

```bash
python scenarionet-converter/inD_scene.py \
  --root_dir /path/to/inD-dataset-v1.1 \
  --segment_size 203 \
  --output_dir /path/to/output
```

**Outputs:**

* ScenarioNet pickle files (ego-aligned)
* `dataset_summary.pkl`
* `dataset_mapping.pkl`

---

### 2. Convert INTERACTION → ScenarioNet

```bash
python scenarionet-converter/interaction_scene.py \
  --root_dir /path/to/INTERACTION \
  --segment_size 200 \
  --output_dir /path/to/output
```

---

### 3. Convert SinD → ScenarioNet

```bash
python scenarionet-converter/sind_scene.py \
  --root_dir /path/to/SinD \
  --segment_size 243 \
  --output_dir /path/to/output
```

**Required Files:**

* `recording_metas.csv`
* `Veh_smoothed_tracks.csv` / `Ped_smoothed_tracks.csv`
* `map.osm`

---

### 4. Convert ScenarioNet → VBD

```bash
python scenarionet-VBD-converter/convert_scenarionet_to_vbd.py \
  --input_dir /path/to/scenarionet \
  --output_dir /path/to/vbd \
  --frame_rate 30.0 \
  --include_raw
```

---

### 5. Convert ScenarioNet → Scenario Dreamer

(Place the script inside Scenario Dreamer’s `scripts/` directory before running.)

```bash
python scenarionet-scenariodreamer-converter/scenarionet_to_scenariodreamer_waymo.py \
  --input_dir /path/to/scenarionet \
  --output_dir /path/to/scenariodreamer \
  --cfg_path cfgs/dataset/waymo_autoencoder_temporal.yaml \
  --train_ratio 0.7 \
  --val_ratio 0.2 \
  --test_ratio 0.1 \
  --seed 0
```

---

## 📚 Citation

If you use this toolkit, please cite the corresponding datasets and their maintainers.

---

Would you like me to make it even more **minimal and aesthetic** (e.g., with emojis for sections, centered title, badges, etc.) for a more polished GitHub look?
