#!/usr/bin/env python3
import os
import math
import csv
import argparse
import pandas as pd
import numpy as np
import pickle
import shutil
import copy
from collections import defaultdict

from numpy.linalg import norm
from shapely.geometry import Polygon
from scipy.interpolate import interp1d
import utm

# ScenarioNet's internal imports
try:
    from metadrive.scenario import ScenarioDescription as SD
    from metadrive.type import MetaDriveType
except ImportError:
    raise ImportError("Please install ScenarioNet / MetaDrive environment to use these imports.")

# ===========================================================================
# 1. Basic Utility/Mapping (unchanged)
# ===========================================================================
AGENT_TYPE_MAPPING = {
    'car': MetaDriveType.VEHICLE,
    'bus': MetaDriveType.VEHICLE,
    'truck': MetaDriveType.VEHICLE,
    'pedestrian': MetaDriveType.PEDESTRIAN,
    'bicycle': MetaDriveType.CYCLIST,
    'motorcycle': MetaDriveType.CYCLIST,
    'tricycle': MetaDriveType.CYCLIST,
}

import math

def lonlat_to_local(lat, lon, lat0, lon0):

    # metres per degree of latitude on the WGS-84 ellipsoid
    DEG_TO_M_LAT = 111_132.954

    # metres per degree of longitude varies with latitude
    cos_lat0 = math.cos(math.radians(lat0))
    DEG_TO_M_LON = 111_319.459 * cos_lat0

    dx = (lon - lon0) * DEG_TO_M_LON   # east-west offset
    dy = (lat - lat0) * DEG_TO_M_LAT   # north-south offset
    return dx, dy



def map_osm_to_md_type(osm_type, subtype=None):
    """Comprehensive mapping from OSM type/subtype to MetaDriveType for SinD dataset."""
    if osm_type == 'lanelet':
        return MetaDriveType.LANE_SURFACE_STREET
    elif osm_type == 'line_thin':
        if subtype == 'solid':
            return MetaDriveType.LINE_SOLID_SINGLE_WHITE
        elif subtype == 'dashed':
            return MetaDriveType.LINE_BROKEN_SINGLE_WHITE
        else:
            return MetaDriveType.LINE_SOLID_SINGLE_WHITE
    elif osm_type == 'line_thick':
        if subtype == 'solid':
            return MetaDriveType.LINE_SOLID_DOUBLE_WHITE
        elif subtype == 'dashed':
            return MetaDriveType.LINE_BROKEN_SINGLE_WHITE
        else:
            return MetaDriveType.LINE_SOLID_DOUBLE_WHITE
    elif osm_type == 'guard_rail':
        return MetaDriveType.GUARDRAIL
    elif osm_type == 'zebra':
        return MetaDriveType.CROSSWALK
    elif osm_type == 'zebra_marking':
        # SinD's version of crosswalks
        return MetaDriveType.CROSSWALK
    elif osm_type == 'stop_line':
        # Stop lines are thick road markings
        return MetaDriveType.LINE_SOLID_SINGLE_WHITE
    elif osm_type == 'curbstone':
        return MetaDriveType.BOUNDARY_LINE
    elif osm_type == 'virtual':
        # Virtual lines are usually lane dividers
        return MetaDriveType.LINE_BROKEN_SINGLE_WHITE
    elif osm_type == 'traffic_light':
        # Traffic lights are not map features, skip them
        return None
    elif osm_type == 'origin':
        return None
    else:
        # Unknown types should be skipped, not mapped to LINE_UNKNOWN
        return None

def are_boundaries_aligned(left_coords, right_coords):
    """Check if left/right boundary directions are consistent; reverse if needed."""
    left_dir = np.array(left_coords[-1]) - np.array(left_coords[0])
    right_dir = np.array(right_coords[-1]) - np.array(right_coords[0])
    dot_product = np.dot(left_dir, right_dir)
    return dot_product >= 0

def resample_coords(coords, num_points):
    """Uniformly resample a list of points to `num_points` via linear interpolation."""
    if len(coords) < 2:
        return np.array(coords)
    distance = np.cumsum([0] + [
        norm(np.array(coords[i]) - np.array(coords[i-1]))
        for i in range(1, len(coords))
    ])
    if distance[-1] <= 0:
        return np.array(coords)
    distance /= distance[-1]
    interpolator = interp1d(distance, np.array(coords), axis=0, kind='linear')
    new_distances = np.linspace(0, 1, num_points)
    return interpolator(new_distances)

def compute_continuous_valid_length(valid_array):
    max_length = current_length = 0
    for v in valid_array:
        if v:
            current_length += 1
            max_length = max(max_length, current_length)
        else:
            current_length = 0
    return max_length


def compute_lane_direction(polyline, eps=1e-6):
    """Compute a 2D unit direction vector for a lane polyline."""
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

# ===========================================================================
# 2. SinD Reading & Splitting Data (MODIFIED)
# ===========================================================================
def read_sind_data(sind_dir):
    """
    Reads the five SinD CSV files from a SinD subdirectory.
    Expected files:
      - recoding_metas.csv
      - Veh_smoothed_tracks.csv and Veh_tracks_meta.csv
      - Ped_smoothed_tracks.csv and Ped_tracks_meta.csv
    Returns:
      frame_rate, dt, xUtmOrigin, yUtmOrigin, rec_meta_df, tracks_meta, agents
    """
    rec_meta_path = os.path.join(sind_dir, "recoding_metas.csv")
    rec_meta_df = pd.read_csv(rec_meta_path)
    meta_row = rec_meta_df.iloc[0]
    frame_rate = float(meta_row["Raw frame rate"])  # fixed at 29.97hz in SinD
    dt = 1.0 / frame_rate
    xUtmOrigin, yUtmOrigin = 0.0, 0.0

    # Vehicles
    veh_meta_path = os.path.join(sind_dir, "Veh_tracks_meta.csv")
    veh_meta_df = pd.read_csv(veh_meta_path)
    veh_tracks_meta = {}
    for _, row in veh_meta_df.iterrows():
        track_id = str(row["trackId"])
        veh_tracks_meta[track_id] = {
            "initialFrame": int(row["initialFrame"]),
            "finalFrame": int(row["finalFrame"]),
            "Frame_nums": int(row["Frame_nums"]),
            "length": float(row["length"]),
            "width": float(row["width"]),
            "agent_type": str(row["class"]).strip(),
            "CrossType": row.get("CrossType", ""),
            "Signal_Violation_Behavior": row.get("Signal_Violation_Behavior", "")
        }
    veh_tracks_path = os.path.join(sind_dir, "Veh_smoothed_tracks.csv")
    veh_tracks_df = pd.read_csv(veh_tracks_path)
    veh_agents = {}
    for _, row in veh_tracks_df.iterrows():
        track_id = str(row["track_id"])
        rec = {
            "frame": int(row["frame_id"]),
            "x": float(row["x"]),
            "y": float(row["y"]),
            "heading": float(row["heading_rad"]),  # already in radians
            "vx": float(row["vx"]),
            "vy": float(row["vy"])
        }
        veh_agents.setdefault(track_id, []).append(rec)

    # Pedestrians
    ped_meta_path = os.path.join(sind_dir, "Ped_tracks_meta.csv")
    ped_meta_df = pd.read_csv(ped_meta_path)
    ped_tracks_meta = {}
    for _, row in ped_meta_df.iterrows():
        track_id = str(row["trackId"])
        ped_tracks_meta[track_id] = {
            "initialFrame": int(row["initialFrame"]),
            "finalFrame": int(row["finalFrame"]),
            "Frame_nums": int(row["Frame_nums"]),
            "agent_type": "pedestrian"
        }
    ped_tracks_path = os.path.join(sind_dir, "Ped_smoothed_tracks.csv")
    ped_tracks_df = pd.read_csv(ped_tracks_path)
    ped_agents = {}
    for _, row in ped_tracks_df.iterrows():
        track_id = str(row["track_id"])
        rec = {
            "frame": int(row["frame_id"]),
            "x": float(row["x"]),
            "y": float(row["y"]),
            "heading": 0.0,
            "vx": float(row["vx"]),
            "vy": float(row["vy"])
        }
        ped_agents.setdefault(track_id, []).append(rec)

    tracks_meta = {}
    agents = {}
    for track_id, meta in veh_tracks_meta.items():
        tracks_meta[track_id] = meta
        agents[track_id] = veh_agents.get(track_id, [])
    for track_id, meta in ped_tracks_meta.items():
        tracks_meta[track_id] = meta
        agents[track_id] = ped_agents.get(track_id, [])

    return frame_rate, dt, xUtmOrigin, yUtmOrigin, rec_meta_df, tracks_meta, agents

import logging
# Setup logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

def process_agents_direct_sind(tracks_meta, agents):
    rows = []
    default_ped_width = 0.3
    default_ped_height = 0.3
    for track_id, records in agents.items():
        records = sorted(records, key=lambda r: r['frame'])
        if track_id in tracks_meta:
            meta = tracks_meta[track_id]
            if meta["agent_type"].lower() == "pedestrian":
                avg_width = default_ped_width
                avg_height = default_ped_height
            else:
                avg_width = meta["width"]
                avg_height = meta["length"]
            agent_type = meta["agent_type"].lower()
        else:
            avg_width, avg_height = 0.0, 0.0
            agent_type = "unknown"
        
        for r in records:

            psi_rad = r['heading'] # Already in radians
            # *** ADD THIS NORMALIZATION ***
            while psi_rad > math.pi:
                psi_rad -= 2 * math.pi
            while psi_rad <= -math.pi:
                psi_rad += 2 * math.pi

            row = {
                'agent_id': track_id,
                'frame_number': r['frame'],
                'agent_type': agent_type,
                'x_position_m': r['x'],
                'y_position_m': r['y'],
                'avg_width_m': avg_width,
                'avg_height_m': avg_height,
                'psi_rad_rad': psi_rad,
                'vx_m_s': r['vx'],
                'vy_m_s': r['vy']
            }
            rows.append(row)
    logging.debug("Processed %d agent rows", len(rows))
    return rows

def split_into_segments(rows, segment_size):
    if not rows:
        return []
    min_frame = min(r['frame_number'] for r in rows)
    max_frame = max(r['frame_number'] for r in rows)
    total_frames = max_frame - min_frame + 1
    num_segments = math.ceil(total_frames / segment_size)
    segments = [[] for _ in range(num_segments)]
    for r in rows:
        seg_index = (r['frame_number'] - min_frame) // segment_size
        seg_index = max(0, min(seg_index, num_segments - 1))
        segments[seg_index].append(r)
    return [seg for seg in segments if seg]

# ===========================================================================
# 3. Map Handling (MODIFIED for SinD)
# ===========================================================================
osm_cache = {}

def parse_osm_map(osm_file, xUtmOrigin, yUtmOrigin):
    """Read OSM, convert nodes to local coords, parse ways/relations to map_features."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(osm_file)
    root = tree.getroot()

    # 1) Build node dict (OSM ID -> local coords)
    nodes = {}
    for node in root.findall('node'):
        node_id = node.attrib['id']
        lat = float(node.attrib['lat'])
        lon = float(node.attrib['lon'])
        local_x, local_y = lonlat_to_local(lat, lon, xUtmOrigin, yUtmOrigin)
        nodes[node_id] = (local_x, local_y)

    # 2) Build ways
    ways = {}
    for way in root.findall('way'):
        way_id = way.attrib['id']
        nd_refs = [nd.attrib['ref'] for nd in way.findall('nd')]
        tags = {tag.attrib['k']: tag.attrib['v'] for tag in way.findall('tag')}
        ways[way_id] = {
            'nd_refs': nd_refs,
            'tags': tags
        }

    # 3) Build relations
    relations = []
    
    for rel in root.findall('relation'):
        rel_id = rel.attrib['id']
        members = []
        for m in rel.findall('member'):
            members.append({
                'type': m.attrib['type'],
                'ref':  m.attrib['ref'],
                'role': m.attrib['role']
            })
        tags = {tag.attrib['k']: tag.attrib['v'] for tag in rel.findall('tag')}
        relations.append({'id': rel_id, 'members': members, 'tags': tags})

    # 4) Convert ways to map_features
    from shapely.geometry import Polygon
    map_features = {}
    for way_id, wdata in ways.items():
        osm_type = wdata['tags'].get('type')
        subtype  = wdata['tags'].get('subtype')
        md_type  = map_osm_to_md_type(osm_type, subtype)
        if md_type is None:
            continue
        coords = [nodes[ref] for ref in wdata['nd_refs'] if ref in nodes]
        if len(coords) < 2:
            continue
            
        coords_array = np.asarray(coords, dtype=float)
        
        # Special handling for crosswalks - they need polygons, not polylines
        if md_type == MetaDriveType.CROSSWALK:
            # For crosswalks, create a rectangular polygon from the line
            if len(coords_array) == 2:
                # Create a rectangle from a 2-point line (pedestrian crossing)
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
                # For multi-point crosswalks, create a polygon by offsetting the line
                coords_2d = coords_array[:, :2]  # Take only x,y
                
                # Calculate average direction for the line
                total_direction = np.array([0.0, 0.0])
                for i in range(len(coords_2d) - 1):
                    seg_dir = coords_2d[i + 1] - coords_2d[i]
                    seg_len = np.linalg.norm(seg_dir)
                    if seg_len > 0:
                        total_direction += seg_dir / seg_len
                
                if np.linalg.norm(total_direction) > 0:
                    total_direction = total_direction / np.linalg.norm(total_direction)
                    # Create perpendicular vector (rotate 90 degrees)
                    perp = np.array([-total_direction[1], total_direction[0]])
                    
                    # Make crosswalk 3 meters wide
                    width = 3.0
                    half_width = width / 2.0
                    
                    # Create offset lines on both sides
                    left_line = coords_2d + perp * half_width
                    right_line = coords_2d - perp * half_width
                    
                    # Build polygon: left line + reversed right line + close
                    coords_array = np.vstack([
                        left_line,
                        right_line[::-1],  # Reverse right line
                        left_line[0:1]     # Close polygon
                    ]).astype(np.float32)
                else:
                    # Fallback: just ensure polygon is closed and 2D
                    if len(coords_2d) > 2 and not np.array_equal(coords_2d[0], coords_2d[-1]):
                        coords_array = np.vstack([coords_2d, coords_2d[0:1]]).astype(np.float32)
                    else:
                        coords_array = coords_2d.astype(np.float32)
            
            map_features[way_id] = {'type': md_type, 'polygon': coords_array}
        else:
            # All other features use polyline
            map_features[way_id] = {'type': md_type, 'polyline': coords_array}

    # 5) Lanelets from relations
    from shapely.geometry   import LineString, MultiLineString, Polygon
    from shapely.ops        import linemerge

    lane_boundary_refs = {}

    for rel in relations:
        if rel['tags'].get('type') != 'lanelet':
            continue

        # 1) Gather all left/right way-IDs
        left_ids  = [m['ref'] for m in rel['members']
                    if m['type']=='way' and m['role']=='left']
        right_ids = [m['ref'] for m in rel['members']
                    if m['type']=='way' and m['role']=='right']
        if not left_ids or not right_ids:
            continue

        # 2) Build and merge Shapely LineStrings
        left_lines  = [ LineString([nodes[n] for n in ways[w]['nd_refs']])
                        for w in left_ids ]
        right_lines = [ LineString([nodes[n] for n in ways[w]['nd_refs']])
                        for w in right_ids ]
        left_merged  = linemerge(MultiLineString(left_lines))
        right_merged = linemerge(MultiLineString(right_lines))

        # Extract coords (handles both LineString and MultiLineString)
        def extract_coords(geom):
            if geom.geom_type == 'LineString':
                return list(geom.coords)
            # if it's still multiple parts, pick the longest
            parts = list(geom)
            longest = max(parts, key=lambda g: g.length)
            return list(longest.coords)

        left_coords  = extract_coords(left_merged)
        right_coords = extract_coords(right_merged)

        # 3) Align orientation of right to left
        if not are_boundaries_aligned(left_coords, right_coords):
            right_coords.reverse()

        # 4) Resample to same number of points
        num_pts = max(len(left_coords), len(right_coords))
        l_res   = resample_coords(left_coords, num_pts)
        r_res   = resample_coords(right_coords, num_pts)

        # 5) Geometric "is-left-on-left?" test & flip centerline if needed
        center = (l_res + r_res) / 2
        def is_left_on_left(center, left, step=5):
            signs = []
            for i in range(0, len(center)-1, step):
                v     = center[i+1] - center[i]
                l_off = left[i] - center[i]
                signs.append(np.sign(v[0]*l_off[1] - v[1]*l_off[0]))
            return np.mean(signs) > 0

        if not is_left_on_left(center, l_res):
            center = center[::-1]
            l_res  = l_res[::-1]
            r_res  = r_res[::-1]

        # 6) Build the full polygon shell (no gaps)
        shell = np.vstack([
            l_res,
            r_res[::-1],
            l_res[0:1]          # close the loop
        ])
        lane_poly = Polygon(shell)

        # 7) Fix invalid geometry, if any
        if not lane_poly.is_valid:
            lane_poly = lane_poly.buffer(0)

        poly_coords = np.array(lane_poly.exterior.coords)

        # 8) Register the lane feature
        lid = f"{rel['id']}"
        map_features[lid] = {
            'type':            MetaDriveType.LANE_SURFACE_STREET,
            'polyline':        center,
            'polygon':         poly_coords,
            'left_boundary':   l_res,
            'right_boundary':  r_res,
            'entry_lanes':     [],
            'exit_lanes':      [],
            'left_neighbor':   [],
            'right_neighbor':  [],
            'speed_limit_kmh': [50],
            'metadata':        rel['tags']
        }
        lane_boundary_refs[lid] = {
            'left': left_ids,
            'right': right_ids
        }

    boundary_to_lanes = defaultdict(list)
    for lane_id, bounds in lane_boundary_refs.items():
        for way_id in bounds['left']:
            boundary_to_lanes[way_id].append((lane_id, 'left'))
        for way_id in bounds['right']:
            boundary_to_lanes[way_id].append((lane_id, 'right'))

    lane_direction_cache = {}
    for lane_id, feat in map_features.items():
        if feat.get('type') == MetaDriveType.LANE_SURFACE_STREET and 'polyline' in feat:
            lane_direction_cache[lane_id] = compute_lane_direction(feat['polyline'])
        else:
            lane_direction_cache[lane_id] = None

    left_neighbor_sets = defaultdict(set)
    right_neighbor_sets = defaultdict(set)

    for lane_id, bounds in lane_boundary_refs.items():
        for way_id in bounds['left']:
            for other_lane, side in boundary_to_lanes.get(way_id, []):
                if other_lane == lane_id or side != 'right':
                    continue
                left_neighbor_sets[lane_id].add(other_lane)
                right_neighbor_sets[other_lane].add(lane_id)
        for way_id in bounds['right']:
            for other_lane, side in boundary_to_lanes.get(way_id, []):
                if other_lane == lane_id or side != 'left':
                    continue
                right_neighbor_sets[lane_id].add(other_lane)
                left_neighbor_sets[other_lane].add(lane_id)

    for lane_id, feat in map_features.items():
        if feat.get('type') != MetaDriveType.LANE_SURFACE_STREET:
            continue
        feat['left_neighbor'] = sorted(left_neighbor_sets.get(lane_id, []), key=str)
        feat['right_neighbor'] = sorted(right_neighbor_sets.get(lane_id, []), key=str)

    # 6) Connect lane endpoints
    lane_endpoints = {}
    angle_cos_threshold = math.cos(math.radians(45.0))
    for mf_id, feat in map_features.items():
        if feat['type'] == MetaDriveType.LANE_SURFACE_STREET and 'polyline' in feat:
            pl = feat['polyline']
            if len(pl) >= 2:
                start_vec = pl[1, :2] - pl[0, :2]
                end_vec = pl[-1, :2] - pl[-2, :2]
                start_norm = np.linalg.norm(start_vec)
                end_norm = np.linalg.norm(end_vec)
                lane_endpoints[mf_id] = {
                    'start': pl[0],
                    'end': pl[-1],
                    'start_dir': start_vec / start_norm if start_norm > 1e-6 else None,
                    'end_dir': end_vec / end_norm if end_norm > 1e-6 else None,
                }

    dist_thr = 2.0
    for lane_id, se in lane_endpoints.items():
        start_pt = se['start']
        end_pt   = se['end']
        entry_lanes=[]
        exit_lanes=[]
        for other_id, ose in lane_endpoints.items():
            if other_id == lane_id:
                continue
            if np.linalg.norm(start_pt - ose['end']) < dist_thr:
                aligned = True
                if se.get('start_dir') is not None and ose.get('end_dir') is not None:
                    dot = np.dot(se['start_dir'], ose['end_dir'])
                    aligned = dot > angle_cos_threshold
                if aligned and other_id not in entry_lanes:
                    entry_lanes.append(other_id)
            if np.linalg.norm(end_pt - ose['start']) < dist_thr:
                aligned = True
                if se.get('end_dir') is not None and ose.get('start_dir') is not None:
                    dot = np.dot(se['end_dir'], ose['start_dir'])
                    aligned = dot > angle_cos_threshold
                if aligned and other_id not in exit_lanes:
                    exit_lanes.append(other_id)
        map_features[lane_id]['entry_lanes'] = entry_lanes
        map_features[lane_id]['exit_lanes']  = exit_lanes

    return map_features, (0,0)

def get_sind_map(osm_file, xUtmOrigin, yUtmOrigin):
    if not os.path.exists(osm_file):
        print(f"[WARN] OSM file {osm_file} not found; returning empty map features.")
        return {}, (0, 0)
    map_features, map_center = parse_osm_map(osm_file, xUtmOrigin, yUtmOrigin)
    return map_features, map_center

# ===========================================================================
# 4. create_scenario_from_csv (mostly unchanged)
# ===========================================================================

def create_scenario_from_csv(scenario_data, map_features, map_center, scenario_id,
                             dataset_version, xUtmOrigin, yUtmOrigin):
    scenario = SD()
    scenario[SD.ID] = scenario_id
    scenario[SD.VERSION] = dataset_version
    scenario[SD.METADATA] = {}
    scenario[SD.METADATA][SD.COORDINATE] = "right-handed"
    scenario[SD.METADATA]["dataset"] = "SinD"
    scenario[SD.METADATA]["scenario_id"] = scenario_id
    scenario[SD.METADATA]["metadrive_processed"] = False
    scenario[SD.METADATA]["map"] = "utm_projection"
    scenario[SD.METADATA]["date"] = "2025-01-01"
    scenario[SD.METADATA]['id'] = scenario_id

    sample_rate = 1.0 / 29.97
    scenario[SD.METADATA]["sample_rate"] = sample_rate
    time_step = sample_rate

    scenario[SD.MAP_FEATURES] = map_features
    frames = sorted(set(int(r['frame_number']) for r in scenario_data))
    num_frames = len(frames)
    scenario[SD.LENGTH] = num_frames

    scenario[SD.METADATA][SD.TIMESTEP] = np.linspace(0, (num_frames - 1) * time_step, num_frames)
    scenario[SD.TIMESTEP] = scenario[SD.METADATA][SD.TIMESTEP]
    frame_to_idx = {f: i for i, f in enumerate(frames)}

    # Group by agent
    from collections import defaultdict
    agent_dict = defaultdict(list)
    agent_types = {}
    for r in scenario_data:
        agent_id = r['agent_id']
        agent_dict[agent_id].append(r)
        agent_types[agent_id] = r['agent_type'].lower()

    scenario[SD.TRACKS] = {}
    object_summary = {}

    for agent_id, recs in agent_dict.items():
        recs = sorted(recs, key=lambda x: int(x['frame_number']))
        positions = np.zeros((num_frames, 3))
        headings = np.zeros(num_frames)
        velocities = np.zeros((num_frames, 2))
        lengths = np.zeros((num_frames, 1))
        widths = np.zeros((num_frames, 1))
        heights = np.zeros((num_frames, 1))
        valid = np.zeros(num_frames)

        for rec in recs:
            fn = int(rec['frame_number'])
            idx = frame_to_idx[fn]
            positions[idx, 0] = float(rec['x_position_m'])
            positions[idx, 1] = float(rec['y_position_m'])
            headings[idx] = float(rec['psi_rad_rad'])
            velocities[idx, 0] = float(rec['vx_m_s'])
            velocities[idx, 1] = float(rec['vy_m_s'])
            lengths[idx, 0] = float(rec['avg_height_m'])
            widths[idx, 0] = float(rec['avg_width_m'])
            heights[idx, 0] = 1.5
            valid[idx] = 1

        raw_type = agent_types[agent_id]
        agent_type = AGENT_TYPE_MAPPING.get(raw_type, MetaDriveType.OTHER)

        if len(positions[valid > 0]) >= 2:
            deltas = np.diff(positions[valid > 0][:, :2], axis=0)
            dist = np.sum(np.linalg.norm(deltas, axis=1))
        else:
            dist = 0.0

        valid_idx = np.where(valid > 0)[0]
        valid_headings = headings[valid_idx]
        if len(valid_headings) >= 2:
            heading_diff = np.diff(valid_headings)
            total_heading_change = np.sum(np.abs(heading_diff))
        else:
            total_heading_change = 0.0

        challenge_score = dist + total_heading_change
        valid_length = int(np.sum(valid))
        cval_len = compute_continuous_valid_length(valid)

        object_summary[agent_id] = {
            'type': agent_type,
            'valid_length': valid_length,
            'continuous_valid_length': cval_len,
            'track_length': num_frames,
            'moving_distance': dist,
            'total_heading_change': total_heading_change,
            'challenge_score': challenge_score,
            'object_id': agent_id
        }
        scenario[SD.TRACKS][agent_id] = {
            SD.TYPE: agent_type,
            SD.STATE: {
                'position': positions,
                'heading': headings,
                'velocity': velocities,
                'length': lengths,
                'width': widths,
                'height': heights,
                'valid': valid,
            },
            SD.METADATA: {
                'track_length': valid_length,
                'type': agent_type,
                'object_id': agent_id,
                'original_id': agent_id
            }
        }
    scenario[SD.METADATA]['object_summary'] = object_summary

    # Log summary information for debugging
    logging.debug("Scenario %s: %d frames, %d objects", scenario_id, num_frames, len(scenario[SD.TRACKS]))

    # Pick exactly one ego: choose the agent with the longest continuous_valid_length.
    fallback_id = max(object_summary, key=lambda aid: object_summary[aid]['continuous_valid_length'])
    valuable_ids = [fallback_id]

    # --- build exactly one scenario variant per vehicle in valuable_ids ---
    scenario_variants = []
    for agent_id in valuable_ids:
        sc_copy = copy.deepcopy(scenario)
        # relabel the chosen vehicle to 'ego'
        if agent_id != 'ego':
            sc_copy[SD.TRACKS]['ego'] = sc_copy[SD.TRACKS].pop(agent_id)
            sc_copy[SD.TRACKS]['ego'][SD.METADATA][SD.OBJECT_ID] = 'ego'
            sc_copy[SD.TRACKS]['ego'][SD.METADATA]['original_id'] = agent_id

        sc_copy[SD.METADATA][SD.SDC_ID] = 'ego'
        sc_copy[SD.METADATA]['tracks_to_predict'] = {
            'ego': {
                'track_id': 'ego',
                'object_type': sc_copy[SD.TRACKS]['ego'][SD.TYPE],
                'difficulty': 0,
                'track_index': list(sc_copy[SD.TRACKS].keys()).index('ego')
            }
        }
        sc_copy[SD.DYNAMIC_MAP_STATES] = {}
        scenario_variants.append(sc_copy)

    for sc in scenario_variants:
        ego = sc[SD.TRACKS]['ego'][SD.STATE]
        # find first valid ego-frame
        first_i   = int(np.where(ego['valid'] > 0)[0][0])
        origin_xy = ego['position'][first_i, :2]

        # shift all map_features
        for feat in sc[SD.MAP_FEATURES].values():
            for k in ('polyline','polygon','left_boundary','right_boundary'):
                if k in feat:
                    feat[k] = feat[k] - origin_xy

        # shift every track's positions
        for tr in sc[SD.TRACKS].values():
            pts = tr[SD.STATE]['position']   # shape (T,3)
            pts[:, :2] -= origin_xy
            tr[SD.STATE]['position'] = pts

        # now compute the ego's initial heading and build a 2×2 rot matrix
        psi0 = ego['heading'][first_i]
        c, s = math.cos(-psi0), math.sin(-psi0)
        R = np.array([[c, -s],
                      [s,  c]], dtype=float)

        # rotate all map features
        for feat in sc[SD.MAP_FEATURES].values():
            for k in ('polyline','polygon','left_boundary','right_boundary'):
                if k in feat:
                    pts = feat[k]  # already translated
                    feat[k] = (R @ pts.T).T

        # rotate every object's positions & velocities
        for tr in sc[SD.TRACKS].values():
            P = tr[SD.STATE]['position']   # (T,3)
            V = tr[SD.STATE]['velocity']   # (T,2)
            P[:,:2] = (R @ P[:,:2].T).T
            V      = (R @ V.T).T
            tr[SD.STATE]['position'][:,0] = P[:,0]
            tr[SD.STATE]['position'][:,1] = P[:,1]
            tr[SD.STATE]['velocity'] = V          # shape (T,2)

        # finally, make the ego's starting yaw zero
        for tr in sc[SD.TRACKS].values():
            
            tr[SD.STATE]['heading'] -= psi0

    return scenario_variants


def save_summary_and_mapping(summary_path, mapping_path, summary, mapping):
    with open(summary_path, 'wb') as f:
        pickle.dump(summary, f)
    with open(mapping_path, 'wb') as f:
        pickle.dump(mapping, f)

def write_scenarios_to_directory(scenarios, output_dir, dataset_name, dataset_version):
    if os.path.exists(output_dir):
        ans = input(f"Output dir {output_dir} exists. Overwrite? (y/n): ")
        if ans.lower() != 'y':
            print("Aborting.")
            return
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    summary = {}
    mapping = {}
    for sc in scenarios:
        sc_id = sc[SD.ID]
        pkl_name = SD.get_export_file_name(dataset_name, dataset_version, sc_id)
        summary[pkl_name] = sc[SD.METADATA]
        mapping[pkl_name] = ""
        sc_dict = sc.to_dict()
        SD.sanity_check(sc_dict)
        pkl_path = os.path.join(output_dir, pkl_name)
        with open(pkl_path, 'wb') as pf:
            pickle.dump(sc_dict, pf)
    summary_path = os.path.join(output_dir, "dataset_summary.pkl")
    mapping_path = os.path.join(output_dir, "dataset_mapping.pkl")
    save_summary_and_mapping(summary_path, mapping_path, summary, mapping)

# ===========================================================================
# 5. MAIN WRAPPER (MODIFIED for SinD)
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Convert SinD data (in multiple subdirectories) => ScenarioNet format."
    )
    parser.add_argument("--root_dir", required=True,
                        help="Path to SinD dataset root, containing subdirectories (e.g. '8_2_1', '8_2_2', ...).")
    parser.add_argument("--segment_size", type=int, default=243,
                        help="Number of frames per scenario segment.")
    parser.add_argument("--output_dir", default=None,
                        help="Where to put final ScenarioNet PKLs. Default: <root_dir>/converted_scenarios")
    args = parser.parse_args()
    root_dir = args.root_dir
    segment_size = args.segment_size
    output_dir = args.output_dir or os.path.join(root_dir, "converted_scenarios")

    # Determine the shared OSM file path from the SinD root folder
    shared_osm_path = os.path.join(root_dir, "map.osm")
    if os.path.exists(shared_osm_path):
        print(f"Using shared OSM file: {shared_osm_path}")
        # Parse the shared OSM file once (using (0,0) as the origin)
        shared_map_features, shared_map_center = get_sind_map(shared_osm_path, 0.0, 0.0)
    else:
        print("No shared OSM file found; map features will be empty.")
        shared_map_features, shared_map_center = {}, (0, 0)

    # Get all subdirectories containing the SinD CSV files.
    subdirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
               if os.path.isdir(os.path.join(root_dir, d))]
    if not subdirs:
        print(f"[ERROR] No subdirectories found in {root_dir}")
        return

    dataset_name = "SinD"
    dataset_version = "1.0"
    all_scenarios = []

    for sind_dir in subdirs:
        print(f"Processing SinD directory: {sind_dir}")
        try:
            (frame_rate, dt, xUtm, yUtm, rec_meta,
             tracks_meta, agents) = read_sind_data(sind_dir)
        except Exception as e:
            print(f"[WARN] Skipping directory {sind_dir} due to error: {e}")
            continue

        rows = process_agents_direct_sind(tracks_meta, agents)
        print(f"Found {len(rows)} rows in {sind_dir}")

        segments = split_into_segments(rows, segment_size)
        if not segments:
            continue

        # Use the shared OSM map for every subdirectory.
        map_features, map_center = shared_map_features, shared_map_center

        for i, seg_data in enumerate(segments, start=1):
            scenario_id = f"{os.path.basename(sind_dir)}_seg{i}"
            scenario_variants = create_scenario_from_csv(
                seg_data,
                map_features, map_center,
                scenario_id,
                dataset_version,
                xUtm, yUtm
            )
            for j, variant in enumerate(scenario_variants, start=1):
                variant_id = f"{scenario_id}_ego_{j}"
                variant[SD.ID] = variant_id
                all_scenarios.append(variant)

    if not all_scenarios:
        print("[INFO] No scenarios were produced.")
        return

    write_scenarios_to_directory(all_scenarios, output_dir, dataset_name, dataset_version)
    print(f"[DONE] Wrote {len(all_scenarios)} scenario PKLs into {output_dir}")

if __name__ == "__main__":
    main()
