#!/usr/bin/env python3
"""
INTERACTION Dataset → ScenarioNet (MetaDrive) Converter
======================================================
This script converts one folder of the INTERACTION dataset (the official
release from UC Berkeley/ETH Zürich) into the **ScenarioNet** pickle format
used by MetaDrive.  Compared with the existing `interaction.py` converter for
inD data, the key differences are:

1.  Track CSV layout
    •  Files are called `vehicle_tracks_XXX.csv` and `pedestrian_tracks_XXX.csv`.
    •  Header for vehicles:  `track_id,frame_id,timestamp_ms,agent_type,x,y,vx,vy,psi_rad,length,width`.
    •  Header for pedestrians: `track_id,frame_id,timestamp_ms,agent_type,x,y,vx,vy`.
2.  Map files
    •  Every scenario folder has an OSM in Lanelet2 format stored once in
       `maps/<SCENARIO>.osm_xy`.  The *xy* flavour already contains **local**
       metric coordinates (attributes `x="..." y="..."`), so we can skip the
       UTM conversion that the inD pipeline needs.
3.  Agent-type mapping – only `car` and `pedestrian/bicycle` appear; the latter
    is force-mapped to `PEDESTRIAN`.

All downstream logic (segmentation, ego-centric alignment, meta-data, sanity
checks, pickle export) stays identical to the inD converter so we re-use most
helper functions verbatim.
"""

import os
import math
import argparse
import pickle
import shutil
from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from numpy.linalg import norm
from shapely.geometry import Polygon
from scipy.interpolate import interp1d
import copy

# ScenarioNet / MetaDrive
try:
    from metadrive.scenario import ScenarioDescription as SD
    from metadrive.type import MetaDriveType
except ImportError as e:  # pragma: no cover
    raise ImportError("Please install ScenarioNet / MetaDrive to use this converter.") from e

# =============================================================================
# 1.  Basic utilities (mostly copied from inD converter)
# =============================================================================

AGENT_TYPE_MAPPING = {
    "car": MetaDriveType.VEHICLE,
    "pedestrian": MetaDriveType.PEDESTRIAN,
    "pedestrian/bicycle": MetaDriveType.PEDESTRIAN,  # ← dataset nomenclature
    "bicycle": MetaDriveType.PEDESTRIAN,  # treat bicycle like pedestrian here
}


def map_osm_to_md_type(osm_type: str, subtype: str = None):
    """Comprehensive mapping OSM → MetaDrive type for INTERACTION dataset."""

    if osm_type == "lanelet":
        return MetaDriveType.LANE_SURFACE_STREET
    elif osm_type == "line_thin":
        if subtype == "solid" or subtype == "solid_solid":
            return MetaDriveType.LINE_SOLID_SINGLE_WHITE
        elif subtype == "dashed":
            return MetaDriveType.LINE_BROKEN_SINGLE_WHITE
        else:
            return MetaDriveType.LINE_SOLID_SINGLE_WHITE
    elif osm_type == "line_thick":
        if subtype == "solid" or subtype == "solid_solid":
            return MetaDriveType.LINE_SOLID_DOUBLE_WHITE
        elif subtype == "dashed":
            return MetaDriveType.LINE_BROKEN_SINGLE_WHITE
        else:
            return MetaDriveType.LINE_SOLID_DOUBLE_WHITE
    elif osm_type == "guard_rail":
        return MetaDriveType.GUARDRAIL
    elif osm_type in {"fence", "wall", "curbstone"}:
        return MetaDriveType.BOUNDARY_LINE  # "ROAD_EDGE_BOUNDARY"
    elif osm_type == "zebra":
        return MetaDriveType.CROSSWALK
    elif osm_type == "pedestrian_marking":
        # INTERACTION's version of crosswalks
        return MetaDriveType.CROSSWALK
    elif osm_type == "stop_line":
        # Stop lines are thick road markings
        return MetaDriveType.LINE_SOLID_SINGLE_WHITE
    elif osm_type == "road_border":
        # Physical road borders
        return MetaDriveType.BOUNDARY_LINE  # "ROAD_EDGE_BOUNDARY"
    elif osm_type == "virtual":
        # Virtual lines are usually lane dividers
        return MetaDriveType.LINE_BROKEN_SINGLE_WHITE
    elif osm_type == "traffic_sign":
        # Traffic signs are not map features, skip them
        return None
    elif osm_type == "origin":  # helper node in some .osm files
        return None
    else:
        # Unknown types should be skipped, not mapped to LINE_UNKNOWN
        return None


# -----------------------------------------------------------------------------
# Geometry helpers (unchanged)
# -----------------------------------------------------------------------------

def are_boundaries_aligned(left_coords, right_coords):
    left_dir = np.array(left_coords[-1]) - np.array(left_coords[0])
    right_dir = np.array(right_coords[-1]) - np.array(right_coords[0])
    return np.dot(left_dir, right_dir) >= 0


def resample_coords(coords, num_points):
    if len(coords) < 2:
        return np.array(coords)

    distance = np.cumsum([0] + [
        norm(np.array(coords[i]) - np.array(coords[i - 1])) for i in range(1, len(coords))
    ])
    if distance[-1] <= 0:
        return np.array(coords)

    distance /= distance[-1]
    interpolator = interp1d(distance, np.array(coords), axis=0, kind="linear")
    return interpolator(np.linspace(0, 1, num_points))


def compute_continuous_valid_length(valid_arr: np.ndarray) -> int:
    max_run = cur = 0
    for v in valid_arr:
        if v:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return max_run


def compute_lane_direction(polyline, eps=1e-6):
    """Return 2D unit direction vector for a lane polyline."""
    if polyline is None:
        return None
    pts = np.asarray(polyline)
    if pts.shape[0] < 2:
        return None
    start = pts[0][:2]
    for next_pt in pts[1:]:
        vec = next_pt[:2] - start
        norm = np.linalg.norm(vec)
        if norm > eps:
            return vec / norm
    return None


# =============================================================================
# 2.  Read INTERACTION CSV (single file → list[row])
# =============================================================================

Row = Dict[str, float]


def read_interaction_csv(csv_path: str) -> List[Row]:
    """Parse *one* `vehicle|pedestrian_tracks_XXX.csv` into a list of dicts.

    The returned list contains rows ready for `create_scenario_from_csv()`.
    """

    is_vehicle = os.path.basename(csv_path).startswith("vehicle_")
    df = pd.read_csv(csv_path)

    # Basic sanity check – mandatory columns
    mandatory = {"track_id", "frame_id", "agent_type", "x", "y"}
    if not mandatory.issubset(df.columns):
        raise ValueError(f"{csv_path}: missing required columns {mandatory - set(df.columns)}")

    rows: List[Row] = []
    for _, r in df.iterrows():
        track_id = str(r["track_id"]).strip()
        agent_type = str(r["agent_type"]).strip().lower()

        # Map special naming
        if agent_type == "pedestrian/bicycle":
            agent_type = "pedestrian"  # treat as pedestrian

        x = float(r["x"])
        y = float(r["y"])
        vx = float(r.get("vx", 0.0))
        vy = float(r.get("vy", 0.0))
        length = float(r.get("length", 0.8 if agent_type == "pedestrian" else 4.0))
        width = float(r.get("width", 0.5 if agent_type == "pedestrian" else 1.8))

        if is_vehicle:
            psi = float(r.get("psi_rad", 0.0))
        else:
            speed = math.hypot(vx, vy)
            psi = math.atan2(vy, vx) if speed > 1e-2 else 0.0

        rows.append({
            "agent_id": track_id,
            "frame_number": int(r["frame_id"]),
            "agent_type": agent_type,
            "x_position_m": x,
            "y_position_m": y,
            "avg_width_m": width,
            "avg_height_m": length,
            "psi_rad_rad": psi,
            "vx_m_s": vx,
            "vy_m_s": vy,
        })

    return rows


# =============================================================================
# 3.  Parse .osm or .osm_xy (local xy → MetaDrive map_features)
# =============================================================================

osm_cache = {}


def parse_osm_generic(osm_path: str):
    """Parse Lanelet2 OSM (.osm **or** .osm_xy with local metric x/y).

    The implementation is a trimmed version of the inD utility, with the only
    change being that we directly read `x=".." y=".."` when present – no UTM
    conversion required.
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(osm_path)
    root = tree.getroot()

    # 1) Nodes
    nodes: Dict[str, Tuple[float, float]] = {}
    for nd in root.findall("node"):
        nid = nd.attrib["id"]
        if "x" in nd.attrib and "y" in nd.attrib:  # .osm_xy format
            nodes[nid] = (float(nd.attrib["x"]), float(nd.attrib["y"]))
        else:  # regular lat/lon – fallback to zero-origin UTM
            lat = float(nd.attrib["lat"])
            lon = float(nd.attrib["lon"])
            import utm  # local import to keep dependency optional

            easting, northing, *_ = utm.from_latlon(lat, lon)
            nodes[nid] = (easting, northing)

    # 2) Ways
    ways = {}
    for way in root.findall("way"):
        wid = way.attrib["id"]
        nd_refs = [n.attrib["ref"] for n in way.findall("nd")]
        tags = {t.attrib["k"]: t.attrib["v"] for t in way.findall("tag")}
        ways[wid] = {"nd_refs": nd_refs, "tags": tags}

    # 3) Relations
    relations = []
    for rel in root.findall("relation"):
        members = [{
            "type": m.attrib["type"],
            "ref": m.attrib["ref"],
            "role": m.attrib.get("role", ""),
        } for m in rel.findall("member")]
        tags = {t.attrib["k"]: t.attrib["v"] for t in rel.findall("tag")}
        relations.append({"id": rel.attrib["id"], "members": members, "tags": tags})

    # 4) map_features from ways
    map_features = {}
    for wid, info in ways.items():
        osm_type = info["tags"].get("type")
        subtype = info["tags"].get("subtype")
        md_type = map_osm_to_md_type(osm_type, subtype)
        if md_type is None:
            continue
        coords = [nodes[n] for n in info["nd_refs"] if n in nodes]
        if len(coords) < 2:
            continue
        
        coords_array = np.asarray(coords, dtype=float)
        
        # Special handling for crosswalks - they need polygons, not polylines
        if md_type == MetaDriveType.CROSSWALK:
            # For crosswalks, create a rectangular polygon from the line
            if len(coords_array) == 2:
                # Create a rectangle from a line (pedestrian crossing)
                p1, p2 = coords_array[0], coords_array[1]
                # Calculate perpendicular direction (using only x,y)
                direction = p2[:2] - p1[:2]
                length = np.linalg.norm(direction)
                if length > 0:
                    direction = direction / length
                    # Create perpendicular vector (rotate 90 degrees)
                    perp = np.array([-direction[1], direction[0]])
                    # Make crosswalk 3 meters wide
                    width = 3.0
                    half_width = width / 2.0
                    # Create rectangle corners (2D only for crosswalks)
                    coords_array = np.array([
                        p1[:2] + perp * half_width,
                        p2[:2] + perp * half_width,
                        p2[:2] - perp * half_width,
                        p1[:2] - perp * half_width,
                        p1[:2] + perp * half_width  # Close the polygon
                    ], dtype=np.float32)
            else:
                # For crosswalks with more points, ensure polygon is closed and 2D
                coords_array = coords_array[:, :2]  # Take only x,y
                if len(coords_array) > 2 and not np.array_equal(coords_array[0], coords_array[-1]):
                    coords_array = np.vstack([coords_array, coords_array[0:1]])
            map_features[wid] = {"type": md_type, "polygon": coords_array}
        else:
            # All other features use polyline
            map_features[wid] = {"type": md_type, "polyline": coords_array}

    # 5) Lanelets via relations (same logic as inD converter)
    from shapely.geometry import LineString, MultiLineString
    from shapely.ops import linemerge

    lane_boundary_refs = {}

    for rel in relations:
        if rel["tags"].get("type") != "lanelet":
            continue
        left_ids = [m["ref"] for m in rel["members"] if m["type"] == "way" and m["role"] == "left"]
        right_ids = [m["ref"] for m in rel["members"] if m["type"] == "way" and m["role"] == "right"]
        if not left_ids or not right_ids:
            continue

        left_lines = [LineString([nodes[n] for n in ways[w]["nd_refs"] if n in nodes]) for w in left_ids]
        right_lines = [LineString([nodes[n] for n in ways[w]["nd_refs"] if n in nodes]) for w in right_ids]
        left_merged = linemerge(MultiLineString(left_lines))
        right_merged = linemerge(MultiLineString(right_lines))

        def _coords(g):
            if g.geom_type == "LineString":
                return list(g.coords)
            parts = list(g)  # MultiLineString
            return list(max(parts, key=lambda p: p.length).coords)

        left_coords = _coords(left_merged)
        right_coords = _coords(right_merged)
        if not are_boundaries_aligned(left_coords, right_coords):
            right_coords.reverse()

        n_pts = max(len(left_coords), len(right_coords))
        l_res = resample_coords(left_coords, n_pts)
        r_res = resample_coords(right_coords, n_pts)
        center = (l_res + r_res) / 2.0

        # ensure left is actually on the left-hand side of centreline
        def _left_of(center_, left_, step=5):
            signs = []
            for i in range(0, len(center_) - 1, step):
                v = center_[i + 1] - center_[i]
                l_off = left_[i] - center_[i]
                signs.append(np.sign(v[0] * l_off[1] - v[1] * l_off[0]))
            return np.mean(signs) > 0

        if not _left_of(center, l_res):
            center = center[::-1]
            l_res = l_res[::-1]
            r_res = r_res[::-1]

        shell = np.vstack([l_res, r_res[::-1], l_res[0:1]])
        lane_poly = Polygon(shell).buffer(0)  # fixes invalid geometries
        lid = str(rel["id"])
        map_features[lid] = {
            "type": MetaDriveType.LANE_SURFACE_STREET,
            "polyline": center,
            "polygon": np.asarray(lane_poly.exterior.coords),
            "left_boundary": l_res,
            "right_boundary": r_res,
            "entry_lanes": [],
            "exit_lanes": [],
            "left_neighbor": [],
            "right_neighbor": [],
            "speed_limit_kmh": [50],
            "metadata": rel["tags"],
        }
        lane_boundary_refs[lid] = {
            "left": left_ids,
            "right": right_ids,
        }

    boundary_to_lanes = defaultdict(list)
    for lane_id, bounds in lane_boundary_refs.items():
        for way_id in bounds["left"]:
            boundary_to_lanes[way_id].append((lane_id, "left"))
        for way_id in bounds["right"]:
            boundary_to_lanes[way_id].append((lane_id, "right"))

    lane_direction_cache = {}
    for lane_id, feat in map_features.items():
        if feat.get("type") == MetaDriveType.LANE_SURFACE_STREET and "polyline" in feat:
            lane_direction_cache[lane_id] = compute_lane_direction(feat["polyline"])
        else:
            lane_direction_cache[lane_id] = None

    left_neighbor_sets = defaultdict(set)
    right_neighbor_sets = defaultdict(set)

    for lane_id, bounds in lane_boundary_refs.items():
        for way_id in bounds["left"]:
            for other_lane, side in boundary_to_lanes.get(way_id, []):
                if other_lane == lane_id or side != "right":
                    continue
                left_neighbor_sets[lane_id].add(other_lane)
                right_neighbor_sets[other_lane].add(lane_id)
        for way_id in bounds["right"]:
            for other_lane, side in boundary_to_lanes.get(way_id, []):
                if other_lane == lane_id or side != "left":
                    continue
                right_neighbor_sets[lane_id].add(other_lane)
                left_neighbor_sets[other_lane].add(lane_id)

    for lane_id, feat in map_features.items():
        if feat.get("type") != MetaDriveType.LANE_SURFACE_STREET:
            continue
        feat["left_neighbor"] = sorted(left_neighbor_sets.get(lane_id, []), key=str)
        feat["right_neighbor"] = sorted(right_neighbor_sets.get(lane_id, []), key=str)

    # 6) quick connectivity pass (same as inD)
    lane_endpoints = {}
    angle_cos_threshold = math.cos(math.radians(45.0))
    for fid, feat in map_features.items():
        if feat["type"] == MetaDriveType.LANE_SURFACE_STREET and "polyline" in feat:
            pl = feat["polyline"]
            if len(pl) >= 2:
                start_vec = pl[1, :2] - pl[0, :2]
                end_vec = pl[-1, :2] - pl[-2, :2]
                start_norm = np.linalg.norm(start_vec)
                end_norm = np.linalg.norm(end_vec)
                lane_endpoints[fid] = {
                    "start": pl[0],
                    "end": pl[-1],
                    "start_dir": start_vec / start_norm if start_norm > 1e-6 else None,
                    "end_dir": end_vec / end_norm if end_norm > 1e-6 else None,
                }

    thr = 2.0
    for lid, se in lane_endpoints.items():
        entries, exits = [], []
        for oid, ose in lane_endpoints.items():
            if oid == lid:
                continue
            if np.linalg.norm(se["start"] - ose["end"]) < thr:
                aligned = True
                if se.get("start_dir") is not None and ose.get("end_dir") is not None:
                    dot = np.dot(se["start_dir"], ose["end_dir"])
                    aligned = dot > angle_cos_threshold
                if aligned and oid not in entries:
                    entries.append(oid)
            if np.linalg.norm(se["end"] - ose["start"]) < thr:
                aligned = True
                if se.get("end_dir") is not None and ose.get("start_dir") is not None:
                    dot = np.dot(se["end_dir"], ose["start_dir"])
                    aligned = dot > angle_cos_threshold
                if aligned and oid not in exits:
                    exits.append(oid)
        map_features[lid]["entry_lanes"] = entries
        map_features[lid]["exit_lanes"] = exits

    return map_features, (0.0, 0.0)  # center ignored; will be re-centred later


def get_map_features(scenario_name: str, maps_dir: str):
    if scenario_name in osm_cache:
        return osm_cache[scenario_name]

    candidates = [os.path.join(maps_dir, f"{scenario_name}.osm_xy"),
                  os.path.join(maps_dir, f"{scenario_name}.osm")]
    osm_path = next((p for p in candidates if os.path.isfile(p)), None)
    if osm_path is None:
        raise FileNotFoundError(f"No .osm[_xy] map found for scenario '{scenario_name}'.")

    mf, mc = parse_osm_generic(osm_path)
    osm_cache[scenario_name] = (mf, mc)
    return mf, mc


# =============================================================================
# 4.  Scenario building (mostly identical to inD)
# =============================================================================


def create_scenario_from_csv(
    scenario_rows: List[Row],
    map_features: dict,
    map_center: Tuple[float, float],
    scenario_id: str,
    dataset_version: str,
    sample_rate: float = 0.1,
):
    """Convert raw per-frame rows → ScenarioDescription variants (ego aligned)."""

    scenario = SD()
    scenario[SD.ID] = scenario_id
    scenario[SD.VERSION] = dataset_version
    scenario[SD.METADATA] = {
        SD.COORDINATE: "right-handed",
        "dataset": "INTERACTION",
        "scenario_id": scenario_id,
        "metadrive_processed": False,
        "map": f"{scenario_id.split('_')[0]}.osm_xy",  # original lanelet file name
        "date": "2025-01-01",  # placeholder
        "sample_rate": sample_rate,
    }

    scenario[SD.MAP_FEATURES] = map_features

    frames = sorted({int(r["frame_number"]) for r in scenario_rows})
    num_frames = len(frames)
    scenario[SD.LENGTH] = num_frames

    timestep_arr = np.linspace(0, (num_frames - 1) * sample_rate, num_frames)
    scenario[SD.METADATA][SD.TIMESTEP] = timestep_arr
    scenario[SD.TIMESTEP] = timestep_arr
    frame_to_idx = {f: i for i, f in enumerate(frames)}

    # group by agent
    per_agent: Dict[str, List[Row]] = defaultdict(list)
    agent_types = {}
    for r in scenario_rows:
        aid = r["agent_id"]
        per_agent[aid].append(r)
        agent_types[aid] = r["agent_type"].lower()

    scenario[SD.TRACKS] = {}
    object_summary = {}

    for aid, recs in per_agent.items():
        recs.sort(key=lambda x: int(x["frame_number"]))
        positions = np.zeros((num_frames, 3))
        headings = np.zeros(num_frames)
        velocities = np.zeros((num_frames, 2))
        lengths = np.zeros((num_frames, 1))
        widths = np.zeros((num_frames, 1))
        heights = np.zeros((num_frames, 1))
        valid = np.zeros(num_frames)

        for rec in recs:
            idx = frame_to_idx[int(rec["frame_number"])]
            positions[idx, 0] = rec["x_position_m"]
            positions[idx, 1] = rec["y_position_m"]
            headings[idx] = rec["psi_rad_rad"]
            velocities[idx, 0] = rec["vx_m_s"]
            velocities[idx, 1] = rec["vy_m_s"]
            lengths[idx, 0] = rec["avg_height_m"]
            widths[idx, 0] = rec["avg_width_m"]
            heights[idx, 0] = 1.5
            valid[idx] = 1

        raw_type = agent_types[aid]
        agent_type = AGENT_TYPE_MAPPING.get(raw_type, MetaDriveType.OTHER)

        # distance travelled
        valid_pos = positions[valid > 0][:, :2]
        dist = float(np.sum(np.linalg.norm(np.diff(valid_pos, axis=0), axis=1))) if len(valid_pos) >= 2 else 0.0

        # heading change
        valid_idx = np.where(valid > 0)[0]
        heading_change = float(np.sum(np.abs(np.diff(headings[valid_idx])))) if len(valid_idx) >= 2 else 0.0

        challenge_score = dist + heading_change
        vlen = int(np.sum(valid))
        cval_len = compute_continuous_valid_length(valid)

        object_summary[aid] = {
            "type": agent_type,
            "valid_length": vlen,
            "continuous_valid_length": cval_len,
            "track_length": num_frames,
            "moving_distance": dist,
            "total_heading_change": heading_change,
            "challenge_score": challenge_score,
            "object_id": aid,
        }

        scenario[SD.TRACKS][aid] = {
            SD.TYPE: agent_type,
            SD.STATE: {
                "position": positions,
                "heading": headings,
                "velocity": velocities,
                "length": lengths,
                "width": widths,
                "height": heights,
                "valid": valid,
            },
            SD.METADATA: {
                "track_length": vlen,
                "type": agent_type,
                SD.OBJECT_ID: aid,
                "original_id": aid,
            },
        }

    scenario[SD.METADATA]["object_summary"] = object_summary

    # number summary (same as inD)
    num_summary = {
        "num_objects": len(per_agent),
        "object_types": set(),
        "num_objects_each_type": defaultdict(int),
        "num_moving_objects": 0,
        "num_moving_objects_each_type": defaultdict(int),
        "num_traffic_lights": 0,
        "num_traffic_light_types": set(),
        "num_traffic_light_each_step": {},
        "num_map_features": len(map_features),
        "map_height_diff": float("-inf"),
    }

    for aid in scenario[SD.TRACKS].keys():
        tp = scenario[SD.TRACKS][aid][SD.TYPE]
        num_summary["object_types"].add(tp)
        num_summary["num_objects_each_type"][tp] += 1
        if object_summary[aid]["moving_distance"] > 0:
            num_summary["num_moving_objects"] += 1
            num_summary["num_moving_objects_each_type"][tp] += 1

    scenario[SD.METADATA]["number_summary"] = num_summary

    # choose ego – longest continuous presence
    ego_id = max(object_summary, key=lambda k: object_summary[k]["continuous_valid_length"])
    scenario_variants = []
    for idx, aid in enumerate([ego_id], 1):  # just 1 variant
        sc = copy.deepcopy(scenario)
        if aid != "ego":
            sc[SD.TRACKS]["ego"] = sc[SD.TRACKS].pop(aid)
            sc[SD.TRACKS]["ego"][SD.METADATA][SD.OBJECT_ID] = "ego"
            sc[SD.TRACKS]["ego"][SD.METADATA]["original_id"] = aid
        sc[SD.METADATA][SD.SDC_ID] = "ego"
        sc[SD.METADATA]["tracks_to_predict"] = {
            "ego": {
                "track_id": "ego",
                "object_type": sc[SD.TRACKS]["ego"][SD.TYPE],
                "difficulty": 0,
                "track_index": list(sc[SD.TRACKS].keys()).index("ego"),
            }
        }
        sc[SD.DYNAMIC_MAP_STATES] = {}
        scenario_variants.append(sc)

    # ego-centric transform per variant
    for sc in scenario_variants:
        ego_state = sc[SD.TRACKS]["ego"][SD.STATE]
        first_valid = int(np.where(ego_state["valid"] > 0)[0][0])
        origin_xy = ego_state["position"][first_valid, :2].copy()
        psi0 = float(ego_state["heading"][first_valid])
        c, s = math.cos(-psi0), math.sin(-psi0)
        R = np.array([[c, -s], [s, c]], dtype=float)

        # shift + rotate map
        for feat in sc[SD.MAP_FEATURES].values():
            for key in ("polyline", "polygon", "left_boundary", "right_boundary"):
                if key in feat:
                    pts = feat[key] - origin_xy
                    feat[key] = (R @ pts.T).T

        # shift + rotate all agents
        for tr in sc[SD.TRACKS].values():
            P = tr[SD.STATE]["position"]
            V = tr[SD.STATE]["velocity"]
            P[:, :2] -= origin_xy
            P[:, :2] = (R @ P[:, :2].T).T
            V[:] = (R @ V.T).T
            tr[SD.STATE]["position"] = P
            tr[SD.STATE]["velocity"] = V
            tr[SD.STATE]["heading"] -= psi0

    return scenario_variants


# =============================================================================
# 5.  Main program
# =============================================================================


def main():
    ap = argparse.ArgumentParser("Convert INTERACTION dataset folder into ScenarioNet pickles")
    ap.add_argument("--root_dir", required=True, help="Path to INTERACTION root containing 'data' & 'maps' sub-folders")
    ap.add_argument("--segment_size", type=int, default=200, help="Optional: split each scenario every N frames (0 = no split)")
    ap.add_argument("--output_dir", default=None, help="Where to save .pkl scenarios (default: <root_dir>/converted_scenarios)")
    args = ap.parse_args()

    root_dir = os.path.abspath(args.root_dir)
    data_dir = os.path.join(root_dir, "data")
    maps_dir = os.path.join(root_dir, "maps")
    out_dir = args.output_dir or os.path.join(root_dir, "converted_scenarios")
    segment_size = max(0, args.segment_size)

    if not os.path.isdir(data_dir):
        ap.error(f"'data' folder not found inside {root_dir}")
    if not os.path.isdir(maps_dir):
        ap.error(f"'maps' folder not found inside {root_dir}")

    scenarios = []
    dataset_name = "INTERACTION"
    dataset_version = "1.0"

    for scenario_name in sorted(os.listdir(data_dir)):
        scenario_path = os.path.join(data_dir, scenario_name)
        if not os.path.isdir(scenario_path):
            continue

        print(f"[INFO] Processing scenario folder '{scenario_name}'")
        map_features, map_center = get_map_features(scenario_name, maps_dir)

        # iterate over all track CSVs inside the folder
        csv_files = [f for f in os.listdir(scenario_path) if f.endswith(".csv")]
        for csv_file in sorted(csv_files):
            csv_path = os.path.join(scenario_path, csv_file)
            try:
                rows = read_interaction_csv(csv_path)
            except Exception as e:
                print(f"  [WARN] skipping {csv_file}: {e}")
                continue

            if not rows:
                continue

            # optional segmentation
            def split_rows(rows_, size):
                if size <= 0:
                    return [rows_]
                min_frame = min(r["frame_number"] for r in rows_)
                max_frame = max(r["frame_number"] for r in rows_)
                num_segs = math.ceil((max_frame - min_frame + 1) / size)
                segs = [[] for _ in range(num_segs)]
                for r in rows_:
                    idx = (r["frame_number"] - min_frame) // size
                    segs[idx].append(r)
                return [s for s in segs if s]

            segments = split_rows(rows, segment_size)
            base_id = os.path.splitext(csv_file)[0]  # e.g. vehicle_tracks_000
            for seg_idx, seg_rows in enumerate(segments, 1):
                sc_id = f"{scenario_name}_{base_id}_seg{seg_idx}"
                variant_list = create_scenario_from_csv(seg_rows, map_features, map_center, sc_id, dataset_version)
                for v in variant_list:
                    v_id = f"{sc_id}_ego"
                    v[SD.ID] = v_id
                    scenarios.append(v)

    if not scenarios:
        print("[DONE] nothing converted – no output written")
        return

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    summary = {}
    mapping = {}
    for sc in scenarios:
        pkl_name = SD.get_export_file_name(dataset_name, dataset_version, sc[SD.ID])
        summary[pkl_name] = sc[SD.METADATA]
        mapping[pkl_name] = ""
        with open(os.path.join(out_dir, pkl_name), "wb") as fh:
            pickle.dump(sc.to_dict(), fh)

    with open(os.path.join(out_dir, "dataset_summary.pkl"), "wb") as fh:
        pickle.dump(summary, fh)
    with open(os.path.join(out_dir, "dataset_mapping.pkl"), "wb") as fh:
        pickle.dump(mapping, fh)

    print(f"[DONE] wrote {len(scenarios)} scenarios into {out_dir}")


if __name__ == "__main__":
    main() 
