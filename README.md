# Drone Tool - Autonomous Driving Dataset Converters

A comprehensive toolkit for converting various autonomous driving datasets (inD, INTERACTION, SinD) into standardized formats compatible with ScenarioNet, VBD, and Scenario Dreamer frameworks.

## Overview

This repository contains conversion tools for transforming drone-view traffic datasets into formats suitable for autonomous driving simulation and prediction tasks. The tools handle trajectory data, map features, and agent metadata from multiple dataset sources.

## Features

- **Multiple Dataset Support**: Convert inD, INTERACTION, and SinD datasets
- **Multiple Output Formats**: Support for ScenarioNet, VBD, and Scenario Dreamer formats
- **OSM Map Integration**: Parse and convert OpenStreetMap/Lanelet2 map data
- **Trajectory Processing**: Handle vehicle and pedestrian trajectories with proper coordinate transformations
- **Ego-centric Alignment**: Automatic ego vehicle alignment and coordinate frame transformation
- **Segmentation Support**: Split long recordings into manageable scenario segments

## Project Structure

```
drone-tool/
├── scenarionet-converter/          # Convert drone datasets to ScenarioNet format
│   ├── inD_scene.py               # inD dataset converter
│   ├── interaction_scene.py       # INTERACTION dataset converter
│   └── sind_scene.py              # SinD dataset converter
├── scenarionet-VBD-converter/      # Convert ScenarioNet to VBD format
│   └── convert_scenarionet_to_vbd.py
└── scenarionet-scenariodreamer-converter/  # Convert to Scenario Dreamer format
    └── scenarionet_to_scenariodreamer_waymo.py
```

## Dependencies

### Core Requirements
```
python >= 3.7
numpy
pandas
scipy
shapely
lxml
utm
tqdm
matplotlib
omegaconf
```

### ScenarioNet/MetaDrive Requirements
```
metadrive-simulator
scenarionet
```

Install dependencies:
```bash
pip install numpy pandas scipy shapely lxml utm tqdm matplotlib omegaconf
pip install metadrive-simulator scenarionet
```

## Usage

### 1. inD Dataset Converter

Converts inD (interaction-dataset) traffic recordings to ScenarioNet format.

**Input Format:**
- `XX_recordingMeta.csv`: Recording metadata with frame rate and UTM origin
- `XX_tracks.csv`: Vehicle trajectories
- `XX_tracksMeta.csv`: Track metadata (dimensions, type)
- `maps/lanelets/`: OSM map files with Lanelet2 format

**Usage:**
```bash
python scenarionet-converter/inD_scene.py \
    --root_dir /path/to/inD-dataset-v1.1 \
    --segment_size 203 \
    --output_dir /path/to/output
```

**Parameters:**
- `--root_dir`: Path to inD dataset root containing 'data/' and 'maps/' folders
- `--segment_size`: Number of frames per scenario segment (default: 203)
- `--output_dir`: Output directory for converted scenarios

**Output:**
- ScenarioNet pickle files with ego-aligned coordinates
- `dataset_summary.pkl`: Metadata summary
- `dataset_mapping.pkl`: File mapping information

### 2. INTERACTION Dataset Converter

Converts INTERACTION dataset (UC Berkeley/ETH Zürich) to ScenarioNet format.

**Input Format:**
- `vehicle_tracks_XXX.csv`: Vehicle trajectory data
- `pedestrian_tracks_XXX.csv`: Pedestrian trajectory data
- `maps/*.osm_xy`: Map files with local metric coordinates

**Usage:**
```bash
python scenarionet-converter/interaction_scene.py \
    --root_dir /path/to/INTERACTION \
    --segment_size 200 \
    --output_dir /path/to/output
```

**Key Features:**
- Direct local coordinate parsing (no UTM conversion needed)
- Handles both vehicle and pedestrian tracks
- Automatic agent type mapping

### 3. SinD Dataset Converter

Converts SinD (Singapore inD) dataset to ScenarioNet format.

**Input Format:**
- `recoding_metas.csv`: Recording metadata
- `Veh_smoothed_tracks.csv` / `Veh_tracks_meta.csv`: Vehicle data
- `Ped_smoothed_tracks.csv` / `Ped_tracks_meta.csv`: Pedestrian data
- `map.osm`: Shared OSM map file

**Usage:**
```bash
python scenarionet-converter/sind_scene.py \
    --root_dir /path/to/SinD \
    --segment_size 243 \
    --output_dir /path/to/output
```

**Special Features:**
- Handles multiple subdirectories (e.g., 8_2_1, 8_2_2)
- Uses shared OSM map across all scenarios
- Fixed frame rate at 29.97Hz

### 4. ScenarioNet to VBD Converter

Converts ScenarioNet pickles to VBD (Vectorized Behavior Dataset) format for motion prediction tasks.

**Usage:**
```bash
python scenarionet-VBD-converter/convert_scenarionet_to_vbd.py \
    --input_dir /path/to/scenarionet/pkl/files \
    --output_dir /path/to/vbd/output \
    --frame_rate 30.0 \
    --include_raw
```

**Parameters:**
- `--input_dir`: Directory with ScenarioNet pickle files
- `--output_dir`: Output directory for VBD format
- `--frame_rate`: Source data frame rate in Hz (default: 30.0)
- `--include_raw`: Include original scenario data in output

**Output Features:**
- Fixed sequence lengths: 11 history frames + 81 future frames
- Resampled to 10Hz target frame rate
- Relation matrix computation between agents, polylines, and traffic lights
- Up to 64 objects, 256 polylines, 16 traffic lights per scenario

### 5. ScenarioNet to Scenario Dreamer Converter

Converts ScenarioNet format to Scenario Dreamer autoencoder preprocessing format.

**Usage:**
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

**Parameters:**
- `--input_dir`: ScenarioNet pickle directory
- `--output_dir`: Output directory (train/val/test subfolders created automatically)
- `--cfg_path`: Hydra dataset config path
- `--train_ratio`, `--val_ratio`, `--test_ratio`: Split ratios (must sum to 1.0)
- `--seed`: Random seed for deterministic splitting

**Key Features:**
- Automatic frame rate conversion (30Hz → 10Hz)
- Train/validation/test split with configurable ratios
- Lane graph extraction with connectivity information
- Agent type mapping to Waymo format

## Data Format Specifications

### ScenarioNet Format
The intermediate ScenarioNet format includes:
- **Tracks**: Agent trajectories with state information (position, heading, velocity, dimensions, validity)
- **Map Features**: Lanes, road boundaries, crosswalks with polylines and connectivity
- **Metadata**: Scenario ID, frame rate, SDC (self-driving car) identification, object summaries

### VBD Format
- **agents_history**: Shape (64, 11, 8) - position, heading, velocity, dimensions
- **agents_future**: Shape (64, 81, 5) - future trajectory predictions
- **polylines**: Shape (256, 30, 5) - map polylines with type information
- **relations**: Pairwise spatial relations between all elements
- **agents_type**: Agent classification (vehicle=1, pedestrian=2, cyclist=3)

### Scenario Dreamer Format
- **objects**: List of agent dictionaries with positions, velocities, headings
- **lane_graph**: Dictionary with lane centerlines and connectivity (pre/suc/left/right pairs)
- **av_idx**: Autonomous vehicle index
- **time_stamps**: Frame timestamps

## Coordinate Systems

All converters handle coordinate transformations:
1. **UTM Conversion**: For inD dataset (lat/lon → local meters)
2. **Local Coordinates**: For INTERACTION dataset (already in meters)
3. **Ego-centric Alignment**: Transform all coordinates relative to ego vehicle
4. **Rotation Normalization**: Align ego heading to zero at the initial frame

## Map Processing

The tools support Lanelet2 OSM format with:
- **Lane Extraction**: Center lines, left/right boundaries
- **Connectivity**: Entry/exit lanes, left/right neighbors
- **Feature Types**: Lane surfaces, road lines, boundaries, crosswalks, guard rails
- **Automatic Resampling**: Uniform point distribution along polylines

## Agent Types

Supported agent classifications:
- **Vehicle**: Cars, trucks, buses
- **Pedestrian**: Pedestrians, walkers
- **Cyclist**: Bicycles, motorcycles, tricycles

## Common Parameters

### Segment Size
- **inD**: 203 frames (~6.8 seconds at 30Hz)
- **INTERACTION**: 200 frames (~20 seconds at 10Hz)
- **SinD**: 243 frames (~8.1 seconds at 29.97Hz)

### Frame Rates
- **inD**: 25-30 Hz (variable)
- **INTERACTION**: 10 Hz
- **SinD**: 29.97 Hz (fixed)
- **VBD Target**: 10 Hz
- **Scenario Dreamer**: 10 Hz

## Output Validation

All converters perform sanity checks:
- Scenario structure validation via `ScenarioDescription.sanity_check()`
- Minimum scenario length requirements
- Valid ego vehicle identification
- Coordinate transformation verification

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install metadrive-simulator scenarionet
   ```

2. **OSM File Not Found**
   - Ensure map files are in the correct directory structure
   - Check file naming conventions (e.g., `location4.osm` for inD)

3. **Invalid Scenarios**
   - Check minimum frame requirements (must exceed history length)
   - Verify ego vehicle has valid trajectory at reference frame

4. **Coordinate Misalignment**
   - Verify UTM origins in recording metadata
   - Check map coordinate reference system

## Citation

If you use these tools in your research, please cite the relevant datasets:

**inD Dataset:**
```
@inproceedings{bock2020ind,
  title={The inD dataset: A drone dataset of naturalistic road user trajectories at German intersections},
  author={Bock, Julian and Krajewski, Robert and Moers, Tobias and Runde, Steffen and Vater, Lennart and Eckstein, Lutz},
  booktitle={2020 IEEE Intelligent Vehicles Symposium (IV)},
  year={2020}
}
```

**INTERACTION Dataset:**
```
@inproceedings{zhan2019interaction,
  title={INTERACTION Dataset: An INTERnational, Adversarial and Cooperative moTION Dataset in Interactive Driving Scenarios with Semantic Maps},
  author={Zhan, Wei and Sun, Liting and Wang, Di and others},
  booktitle={arXiv:1910.03088},
  year={2019}
}
```

**ScenarioNet:**
```
@article{li2023scenarionet,
  title={ScenarioNet: Open-Source Platform for Large-Scale Traffic Scenario Simulation and Modeling},
  author={Li, Quanyi and Peng, Zhenghao and Feng, Lan and others},
  journal={Advances in Neural Information Processing Systems},
  year={2023}
}
```

## License

Please refer to the original dataset licenses:
- inD Dataset: Academic use only
- INTERACTION Dataset: Academic use with attribution
- SinD Dataset: Check with dataset providers

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing style conventions
- All converters maintain ScenarioNet compatibility
- Proper error handling and validation
- Documentation for new features

## Contact

For issues or questions about these conversion tools, please open an issue on GitHub or contact the maintainers.
