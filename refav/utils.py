import numpy as np
import pyvista as pv
import os
from pathlib import Path
from typing import Union, Callable, Any, Literal
import matplotlib
import matplotlib.pyplot as plt
from pathos.multiprocessing import ProcessingPool as Pool
import scipy.ndimage
from scipy.spatial.transform import Rotation
from copy import deepcopy
import vtk
from functools import wraps
import pandas as pd
import inspect
import scipy
import sys
import json
from collections import OrderedDict
from refav.paths import AV2_DATA_DIR

from av2.structures.cuboid import Cuboid, CuboidList
from av2.map.map_api import ArgoverseStaticMap
from av2.map.lane_segment import LaneSegment
from av2.map.pedestrian_crossing import PedestrianCrossing
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.geometry.se3 import SE3
from av2.utils.io import read_feather, read_city_SE3_ego
import av2.geometry.polyline_utils as polyline_utils
import av2.rendering.vector as vector_plotting_utils
from av2.structures.sweep import Sweep
from av2.evaluation.tracking.utils import save
from av2.datasets.sensor.splits import TEST, TRAIN, VAL


class CacheManager:
    def __init__(self, max_caches=50):
        self.caches = {}
        self.stats = {}

        self.max_caches=max_caches
        self.cache_usage = []  # Track cache usage for LRU eviction
        self.num_processes = max(1, int(.9*os.cpu_count()))

    def make_hashable(self, obj):
        if isinstance(obj, (list, tuple, set)):
            return tuple(self.make_hashable(x) for x in obj)
        elif isinstance(obj, dict):
            return tuple(sorted((k, self.make_hashable(v)) for k, v in obj.items()))
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, np.ndarray):
            return tuple(obj.flatten())
        elif isinstance(obj, ArgoverseStaticMap):
            return obj.log_id
        elif isinstance(obj, LaneSegment):
            return obj.id
        else:
            return obj

    def create_cache(self, name, maxsize=128):

        if name not in self.caches and len(self.caches) >= self.max_caches:
            # Evict least recently used cache
            self._evict_least_used_cache()

        if name not in self.caches:
            self.caches[name] = OrderedDict()
            self.stats[name] = {'hits': 0, 'misses': 0}
        else:
            # Move this cache to the end (most recently used)
            self.cache_usage.remove(name)
            self.cache_usage.append(name)

        
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                key = (
                    self.make_hashable(args),
                    self.make_hashable(kwargs)
                )
                
                cache:OrderedDict = self.caches[name]
                
                if key in cache:
                    cache.move_to_end(key)
                    self.stats[name]['hits'] += 1
                    return cache[key]
                
                result = func(*args, **kwargs)
                self.stats[name]['misses'] += 1

                cache[key] = result
                if len(cache) > maxsize:
                    cache.popitem(last=False)
                    
                return result
            
            wrapper.clear_cache = lambda: self.caches[name].clear()
            wrapper.cache_info = lambda: {
                'name': name,
                'current_size': len(self.caches[name]),
                'maxsize': maxsize
            }
            
            return wrapper
        return decorator
    
    def _evict_least_used_cache(self):
        """Evict the least recently used cache"""
        if self.cache_usage:
            oldest_cache = self.cache_usage.pop(0)
            self.caches.pop(oldest_cache)
            self.stats.pop(oldest_cache)
            print(f"Evicted cache: {oldest_cache}")
    
    def clear_all(self):
        for cache in self.caches.values():
            cache.clear()
    
    def info(self):
        return {name: len(cache) for name, cache in self.caches.items()}
    
    def get_stats(self, name=None):
        if name:
            stats = self.stats[name]
            total = stats['hits'] + stats['misses']
            hit_rate = stats['hits'] / total if total > 0 else 0
            return {
                'name': name,
                'hits': stats['hits'],
                'misses': stats['misses'],
                'hit_rate': f"{hit_rate:.2%}",
                'cache_size': len(self.caches[name])
            }
        return {
            name: self.get_stats(name) for name in self.stats
        }

cache_manager = CacheManager()

def composable(composable_func):
    """
    A decorator to evaluate track crossings in parallel for the given composable function.
    
    Args:
        composable_func (function): A function that is evaluated on the track and candidate data.
    
    Returns:
        function: A new function that wraps `composable_func` and adds parallel evaluation.
    """
    @wraps(composable_func)
    def wrapper(track_candidates, log_dir, *args, **kwargs):
        """
        The wrapper function that adds parallel processing and filtering to the decorated function.
        
        Args:
            tracks (dict): Keys are track UUIDs, values are lists of valid timestamps.
            candidates (dict): Keys are candidate UUIDs, values are lists of valid timestamps.
            log_dir (Path): Directory containing log data.
            *args, **kwargs: Additional arguments passed to `composable_func`.
            
        Returns:
            dict: Subset of `track_dict` containing tracks being crossed and their crossing timestamps.
            dict: Nested dict where keys are track UUIDs, values are dicts of candidate UUIDs with their crossing timestamps.
        """
        # Process tracks and candidates into dictionaries
        track_dict = to_scenario_dict(track_candidates, log_dir)

        # Parallelize processing of the UUIDs
        all_uuids = list(track_dict.keys())

        true_tracks, _ = parallelize_uuids(composable_func, all_uuids, log_dir, *args, **kwargs)
        # Apply filtering
        scenario_dict = {}

        for track_uuid, unfiltered_related_objects in track_dict.items():
            if true_tracks.get(track_uuid, None) is not None:
                prior_related_objects = scenario_at_timestamps(unfiltered_related_objects, get_scenario_timestamps(true_tracks[track_uuid]))
                scenario_dict[track_uuid] = prior_related_objects   

        return scenario_dict

    return wrapper

def composable_relational(composable_func):
    """
    A decorator to evaluate track crossings in parallel for the given composable function.
    
    Args:
        composable_func (function): A function that is evaluated on the track and candidate data.
    
    Returns:
        function: A new function that wraps `composable_func` and adds parallel evaluation.
    """
    @wraps(composable_func)
    def wrapper(track_candidates, related_candidates, log_dir, *args, **kwargs):
        """
        The wrapper function that adds parallel processing and filtering to the decorated function.
        
        Args:
            tracks (dict): Keys are track UUIDs, values are lists of valid timestamps.
            candidates (dict): Keys are candidate UUIDs, values are lists of valid timestamps.
            log_dir (Path): Directory containing log data.
            *args, **kwargs: Additional arguments passed to `composable_func`.
            
        Returns:
            dict: Subset of `track_dict` containing tracks being crossed and their crossing timestamps.
            dict: Nested dict where keys are track UUIDs, values are dicts of candidate UUIDs with their crossing timestamps.
        """
        # Process tracks and candidates into dictionaries
        track_dict = to_scenario_dict(track_candidates, log_dir)
        related_candidate_dict = to_scenario_dict(related_candidates, log_dir)
        track_dict, related_candidate_dict = remove_nonintersecting_timestamps(track_dict, related_candidate_dict)

        # Parallelize processing of the UUIDs
        track_uuids = list(track_dict.keys())
        candidate_uuids = list(related_candidate_dict.keys())

        _, relationship_dict = parallelize_uuids(composable_func, track_uuids, candidate_uuids, log_dir, *args, **kwargs)

        # Apply filtering
        scenario_dict = {track_uuid: {} for track_uuid in relationship_dict.keys()}

        for track_uuid, unfiltered_related_objects in track_dict.items():
            if isinstance(unfiltered_related_objects, dict) and track_uuid in relationship_dict:
                prior_related_objects = scenario_at_timestamps(unfiltered_related_objects, get_scenario_timestamps(relationship_dict[track_uuid]))
                scenario_dict[track_uuid] = prior_related_objects   

        for track_uuid, unfiltered_related_objects in relationship_dict.items():
            for related_uuid, related_timestamps in unfiltered_related_objects.items():
                eligible_timestamps = set(related_timestamps).intersection(get_scenario_timestamps(track_dict[track_uuid]))
                scenario_dict[track_uuid][related_uuid] = scenario_at_timestamps(related_candidate_dict[related_uuid], eligible_timestamps)            

        return scenario_dict

    return wrapper


def scenario_at_timestamps(scenario_dict:dict, kept_timestamps):
    scenario_with_timestamps = deepcopy(scenario_dict)

    if not isinstance(scenario_dict, dict):
        return sorted(list(set(scenario_dict).intersection(kept_timestamps)))

    keys_to_remove = []
    for uuid, relationship in scenario_with_timestamps.items():
        relationship = scenario_at_timestamps(relationship, kept_timestamps)
        scenario_with_timestamps[uuid] = relationship
        
        if len(relationship) == 0:
            keys_to_remove.append(uuid)

    for key in keys_to_remove:
        scenario_with_timestamps.pop(key)

    return scenario_with_timestamps


def remove_nonintersecting_timestamps(dict1:dict[str,list], dict2:dict[str,list]):

    dict1_timestamps = get_scenario_timestamps(dict1)
    dict2_timestamps = get_scenario_timestamps(dict2)

    dict1 = scenario_at_timestamps(dict1, dict2_timestamps)
    dict2 = scenario_at_timestamps(dict2, dict1_timestamps)

    return dict1, dict2

@cache_manager.create_cache('get_ego_uuid')
def get_ego_uuid(log_dir):
    df = read_feather(log_dir / 'sm_annotations.feather')
    ego_df = df[df['category'] == 'EGO_VEHICLE']
    return ego_df['track_uuid'].iloc[0]

@composable_relational
def has_objects_in_relative_direction(
    track_candidates:Union[list, dict],
    related_candidates:Union[list, dict], 
    log_dir:Path, 
    direction:Literal["forward", "backward", "left", "right"], 
    min_number=1, 
    max_number=np.inf, 
    within_distance=np.inf, 
    lateral_thresh=np.inf) -> dict:
    """
    Identifies tracked objects with at least the minimum number of related candidates in the specified direction.
    If the minimum number is met, will create relationships equal to the max_number of closest objects. 

    Args:
        track_candidates: Tracks to analyze (list of UUIDs or scenario dictionary).
        related_candidates: Candidates to check for in direction (list of UUIDs or scenario dictionary).
        log_dir: Path to scenario logs.
        direction: Direction to analyze from the track's point of view ('forward', 'backward', 'left', 'right').
        min_number: Minimum number of objects to identify in the direction per timestamp. Defaults to 1.
        max_number: Maximum number of objects to identify in the direction per timestamp. Defaults to infinity.
        within_distance: Maximum distance for considering an object in the direction. Defaults to infinity.
        lateral_thresh: Maximum lateral distance the related object can be from the sides of the tracked object. Defaults to infinity.

    Returns:
        dict: 
            A scenario dictionary where keys are track UUIDs and values are dictionaries containing related candidate UUIDs 
            and lists of timestamps when the condition is met for that relative direction.

    Example:
        vehicles_with_peds_in_front = has_objects_in_relative_direction(vehicles, pedestrians, log_dir, direction='forward', min_number=2)
    """

    track_uuid = track_candidates
    candidate_uuids = related_candidates

    if track_uuid == get_ego_uuid(log_dir):
        #Ford Fusion dimensions offset from ego_coordinate frame
        track_width = 1
        track_front = 4.877/2 + 1.422
        track_back = 4.877 - (4.877/2 + 1.422)
    else:
        track_cuboid = get_cuboid_from_uuid(track_uuid, log_dir)
        track_width = track_cuboid.width_m/2
        track_front = track_cuboid.length_m/2
        track_back = -track_cuboid.length_m/2

    timestamps_with_objects = []
    objects_in_relative_direction = {}
    in_direction_dict = {}

    for candidate_uuid in candidate_uuids:
        if candidate_uuid == track_uuid:
            continue

        pos, timestamps = get_nth_pos_deriv(candidate_uuid, 0, log_dir, coordinate_frame=track_uuid)

        for i in range(len(timestamps)):

            if direction == 'left' and pos[i, 1]>track_width and (track_back-lateral_thresh<pos[i,0]<track_front+lateral_thresh)  \
            or direction == 'right' and pos[i, 1]<-track_width and (track_back-lateral_thresh<pos[i,0]<track_front+lateral_thresh)\
            or direction == 'forward' and pos[i,0]>track_front and (-track_width-lateral_thresh<pos[i,1]<track_width+lateral_thresh)\
            or direction == 'backward' and pos[i,0]<track_back and (-track_width-lateral_thresh<pos[i,1]<track_width+lateral_thresh):  
                if not in_direction_dict.get(timestamps[i], None):
                    in_direction_dict[timestamps[i]] = []
                in_direction_dict[timestamps[i]].append((candidate_uuid, np.linalg.norm(pos[i])))

    for timestamp, objects in in_direction_dict.items():
        sorted_objects = sorted(objects, key=lambda row: row[1])

        count = 0
        for candidate_uuid, distance in sorted_objects:
            if distance <= within_distance and count < max_number:
                timestamps_with_objects.append(timestamp)
                if not objects_in_relative_direction.get(candidate_uuid, None):
                    objects_in_relative_direction[candidate_uuid] = []
                objects_in_relative_direction[candidate_uuid].append(timestamp)
                count += 1

    if len(list(objects_in_relative_direction.keys())) >= min_number:
        return timestamps_with_objects, objects_in_relative_direction
    else:
        return [], {}


#@cache_manager.create_cache('get_objects_in_relative_direction')
def get_objects_in_relative_direction(
    track_candidates:Union[list, dict],
    related_candidates:Union[list, dict], 
    log_dir:Path, 
    direction:Literal["forward", "backward", "left", "right"], 
    min_number:int=0, 
    max_number:int=np.inf, 
    within_distance=np.inf, 
    lateral_thresh=np.inf)->dict:
    """
    Returns a scenario dictionary of the related candidates that are in the relative direction of the track candidates.
    

    Args:
        track_candidates: Tracks  (list of UUIDs or scenario dictionary).
        related_candidates: Candidates to check for in direction (list of UUIDs or scenario dictionary).
        log_dir: Path to scenario logs.
        direction: Direction to analyze from the track's point of view ('forward', 'backward', 'left', 'right').
        min_number: Minimum number of objects to identify in the direction per timestamp. Defaults to 0.
        max_number: Maximum number of objects to identify in the direction per timestamp. Defaults to infinity.
        within_distance: Maximum distance for considering an object in the direction. Defaults to infinity.
        lateral_thresh: Maximum lateral distance the related object can be from the sides of the tracked object. Lateral distance is 
        distance is the distance from the sides of the object that are parallel to the specified direction. Defaults to infinity.

    Returns:
        dict: 
            A scenario dictionary where keys are track UUIDs and values are dictionaries containing related candidate UUIDs 
            and lists of timestamps when the condition is met for that relative direction.

    Example:
        peds_in_front_of_vehicles = get_objects_in_relative_direction(vehicles, pedestrians, log_dir, direction='forward', min_number=2)
    """
    
    tracked_objects = \
    reverse_relationship(has_objects_in_relative_direction)(track_candidates, related_candidates, log_dir, direction,
        min_number=min_number, max_number=max_number, within_distance=within_distance, lateral_thresh=lateral_thresh)

    return tracked_objects


def get_cuboids_of_category(cuboids: list[Cuboid], category):
    objects_of_category = []
    for cuboid in cuboids:
        if cuboid.category == category:
            objects_of_category.append(cuboid)
    return objects_of_category 


def get_uuids_of_category(log_dir:Path, category:str):
    """
    Returns all uuids from a given category from the log annotations. This method accepts the 
    super classes "ANY" and "VEHICLE".

    Args:
        log_dir: Path to the directory containing scenario logs and data.
        category: the category of objects to return

    Returns: 
        list: the uuids of objects that fall within the category

    Example:
        trucks = get_uuids_of_category(log_dir, category='TRUCK')
    """

    df = read_feather(log_dir / 'sm_annotations.feather')

    if category == 'ANY':
        uuids = df['track_uuid'].unique()
    elif category == 'VEHICLE':

        uuids = []
        vehicle_superclass = ["EGO_VEHICLE","ARTICULATED_BUS","BOX_TRUCK","BUS","LARGE_VEHICLE",
                              "MOTORCYCLE","RAILED_VEHICLE","REGULAR_VEHICLE","SCHOOL_BUS","TRUCK","TRUCK_CAB"]
        
        for vehicle_category in vehicle_superclass:
            category_df = df[df['category'] == vehicle_category]
            uuids.extend(category_df['track_uuid'].unique())
    else:
        category_df = df[df['category'] == category]
        uuids = category_df['track_uuid'].unique()

    return uuids

def has_free_will(track_uuid, log_dir):

    df = read_feather(log_dir / 'sm_annotations.feather')
    category = df[df['track_uuid'] == track_uuid]['category'].iloc[0]
    if category in ['ANIMAL','OFFICIAL_SIGNALER','RAILED_VEHICLE','ARTICULATED_BUS','WHEELED_RIDER','SCHOOL_BUS',
                    'MOTORCYCLIST','TRUCK_CAB','VEHICULAR_TRAILER','BICYCLIST','MOTORCYCLE','TRUCK','BOX_TRUCK','BUS',
                    'LARGE_VEHICLE','PEDESTRIAN','REGULAR_VEHICLE']:
        return True
    else:
        return False


def get_objects_of_category(log_dir, category)->dict:
    """
    Returns all objects from a given category from the log annotations. This method accepts the 
    super-categories "ANY" and "VEHICLE".

    Args:
        log_dir: Path to the directory containing scenario logs and data.
        category: the category of objects to return

    Returns: 
        dict: A scenario dict that where keys are the unique id (uuid) of the object and values 
        are the list of timestamps the object is in view of the ego-vehicle.

    Example:
        trucks = get_objects_of_category(log_dir, category='TRUCK')
    """
    return to_scenario_dict(get_uuids_of_category(log_dir, category), log_dir)


@composable
def is_category(track_uuid, log_dir, category):

    if track_uuid in get_uuids_of_category(log_dir, category):
        non_composable_get_object = unwrap_func(get_object)
        return non_composable_get_object(track_uuid, log_dir)
    else:
        return []


@composable
def get_object(track_uuid, log_dir):

    df = read_feather(log_dir / 'sm_annotations.feather')
    track_df = df[df['track_uuid'] == track_uuid]

    if track_df.empty:
        print(f'Given track_uuid {track_uuid} not in log annotations.')
        return []
    else:
        timestamps = track_df['timestamp_ns']
        return sorted(timestamps)

    
def get_timestamps(track_uuid, log_dir):

    df = read_feather(log_dir / 'sm_annotations.feather')
    track_df = df[df['track_uuid'] == track_uuid]

    if track_df.empty:
        print(f'Given track_uuid {track_uuid} not in log annotations.')
        return []
    else:
        timestamps = track_df['timestamp_ns']
        return sorted(timestamps)


def get_lane_segments(avm: ArgoverseStaticMap, position) -> list[LaneSegment]:
    "Get lane segments object is currently in from city coordinate location"
    lane_segments = []

    candidates = avm.get_nearby_lane_segments(position, 5)
    for ls in candidates:
        if is_point_in_polygon(position[:2], ls.polygon_boundary[:,:2]):
            lane_segments.append(ls)

    return lane_segments


def get_pedestrian_crossings(avm: ArgoverseStaticMap, track_polygon) -> list[PedestrianCrossing]:
    "Get pedestrian crossing that object is currently in from city coordinate location"
    ped_crossings = []

    scenario_crossings = avm.get_scenario_ped_crossings()
    for i, pc in enumerate(scenario_crossings):
        if polygons_overlap(pc.polygon[:,:2], track_polygon[:,:2]):
            ped_crossings.append(pc)

    return ped_crossings


@cache_manager.create_cache('get_scenario_lanes')
def get_scenario_lanes(track_uuid:str, log_dir:Path, avm=None, traj=None, timestamps=None)->dict[float,LaneSegment]:
    """Returns: scenario_lanes as a dict giving lane the object is in keyed by timestamp"""

    if not avm:
        avm = get_map(log_dir)

    if traj is None or timestamps is None:
        traj, timestamps = get_nth_pos_deriv(track_uuid, 0, log_dir)

    scenario_lanes:dict[float, LaneSegment] = {}
    lane_collisions:list[tuple[list[LaneSegment], float]] = []
    for i in range(len(traj)):
        lane_segments = get_lane_segments(avm, traj[i])
        if len(lane_segments) > 1:
            lane_collisions.append((lane_segments,timestamps[i]))
            scenario_lanes[timestamps[i]] = None
        elif lane_segments:
            scenario_lanes[timestamps[i]] = lane_segments[0]
        else: 
            scenario_lanes[timestamps[i]] = None

    future_lane_ids = {ls.id for ls in scenario_lanes.values() if ls}
    for ls_group in lane_collisions:
        correct_ls_found = False

        #If the object travels on a successor in the future, the object also traveld in the lane
        for ls in ls_group[0]:
            if any(successor in future_lane_ids for successor in ls.successors for ls in ls_group[0]):
                scenario_lanes[ls_group[1]] = ls
                correct_ls_found = True
                break

        #The object likely ended the log on an intersection (and therefore the current lane has no successors)
        if not correct_ls_found:
            for ls in ls_group[0]:
                turn_direction = get_turn_direction(ls)
                angular_velocities, timestamps = get_nth_yaw_deriv(track_uuid, 1, log_dir, coordinate_frame='self')
                angular_velocity = angular_velocities[np.where(timestamps == ls_group[1])]
                if (turn_direction == 'LEFT' and angular_velocity > 0.15) \
                or (turn_direction == 'RIGHT' and angular_velocity < -0.15) \
                or (turn_direction == 'STRAIGHT' and 0.15 < angular_velocity < 0.15):
                    scenario_lanes[ls_group[1]] = ls
                    correct_ls_found = True
                    break
        
        #Pick the one of the lanes segments to add to future lanes
        if not correct_ls_found:
            scenario_lanes[ls_group[1]] = ls_group[0][0]

    return scenario_lanes


@cache_manager.create_cache('get_semantic_lane')
def get_semantic_lane(ls: LaneSegment, log_dir, avm=None) -> list[LaneSegment]:
    """Returns a list of lane segments that would make up a single 'lane' coloquailly.
    Finds all lane segments that are directionally forward and backward to the given lane
    segment."""

    if not ls:
        return []

    if not avm:
        avm = get_map(log_dir)
    lane_segments = avm.vector_lane_segments

    try:
        with open('output/misc/semantic_lane_cache.json', 'rb') as file:
            semantic_lane_cache = json.load(file)
            semantic_lanes = semantic_lane_cache[log_dir.name][ls.id]
            all_lanes = avm.vector_lane_segments
            return [all_lanes[ls_id] for ls_id in semantic_lanes]
    except:
        pass

    semantic_lane = [ls]

    if not ls.is_intersection or get_turn_direction(ls) == 'straight':
        predecessors = [ls]
        sucessors = [ls]
    else:
        return semantic_lane

    while predecessors:
        pred_ls = predecessors.pop()
        pred_direction = get_lane_orientation(pred_ls, avm)
        ppred_ids = pred_ls.predecessors
        
        most_likely_pred = None
        best_similarity = 0
        for ppred_id in ppred_ids:
            if ppred_id in lane_segments:
                ppred_ls = lane_segments[ppred_id]
                ppred_direction = get_lane_orientation(ppred_ls, avm)
                similarity = np.dot(ppred_direction, pred_direction)/(np.linalg.norm(ppred_direction)*np.linalg.norm(pred_direction))

                if ((not ppred_ls.is_intersection
                or get_turn_direction(lane_segments[ppred_id]) == 'straight') 
                and similarity > best_similarity):
                    best_similarity = similarity
                    most_likely_pred = ppred_ls

        if most_likely_pred and most_likely_pred not in semantic_lane:
            semantic_lane.append(most_likely_pred)
            predecessors.append(most_likely_pred)

    while sucessors:
        pred_ls = sucessors.pop()
        pred_direction = get_lane_orientation(pred_ls, avm)
        ppred_ids = pred_ls.successors
        
        most_likely_pred = None
        best_similarity = -np.inf
        for ppred_id in ppred_ids:
            if ppred_id in lane_segments:
                ppred_ls = lane_segments[ppred_id]
                ppred_direction = get_lane_orientation(ppred_ls, avm)
                similarity = np.dot(ppred_direction, pred_direction)/(np.linalg.norm(ppred_direction)*np.linalg.norm(pred_direction))

                if ((not ppred_ls.is_intersection
                or get_turn_direction(lane_segments[ppred_id]) == 'straight') 
                and similarity > best_similarity):
                    best_similarity = similarity
                    most_likely_pred = ppred_ls

        if most_likely_pred and most_likely_pred not in semantic_lane:
            semantic_lane.append(most_likely_pred)
            sucessors.append(most_likely_pred)
    
    return semantic_lane


def get_turn_direction(ls: LaneSegment):

    if not ls or not ls.is_intersection:
        return None

    start_direction = ls.right_lane_boundary.xyz[0,:2] - ls.left_lane_boundary.xyz[0,:2]
    end_direction = ls.right_lane_boundary.xyz[-1,:2] - ls.left_lane_boundary.xyz[-1,:2]

    start_angle = np.atan2(start_direction[0], start_direction[1])
    end_angle = np.atan2(end_direction[0], end_direction[1])

    angle_change = end_angle - start_angle

    if abs(angle_change) > np.pi:
        if angle_change > 0:
            angle_change -= 2*np.pi
        else:
            angle_change += 2*np.pi

    if angle_change > np.pi/6:
        return 'right'
    elif angle_change < -np.pi/6:
        return 'left'
    else:
        return 'straight'

def get_lane_orientation(ls: LaneSegment, avm: ArgoverseStaticMap) -> tuple:
    "Returns orientation (as unit direction vectors) at the start and end of the LaneSegment"
    centerline = avm.get_lane_segment_centerline(ls.id)
    orientation  = centerline[-1] - centerline[0]
    return orientation

@composable
def turning(
    track_candidates: Union[list, dict], 
    log_dir:Path,
    direction:Literal["left", "right", None]=None)->dict:
    """
    Returns objects that are turning in the given direction. 

    Args:
        track_candidates: The objects you want to filter from (list of UUIDs or scenario dictionary).
        log_dir: Path to scenario logs.
        direction: The direction of the turn, from the track's point of view ('left', 'right', None).

    Returns:
        dict: 
            A filtered scenario dictionary where:
            - Keys are track UUIDs that meet the turning criteria.
            - Values are nested dictionaries containing timestamps.

    Example:
        turning_left = turning(vehicles, log_dir, direction='left')
    """
    track_uuid = track_candidates

    if direction and direction != 'left' and direction != 'right':
        direction = None
        print("Specified direction must be 'left', 'right', or None. Direction set to \
              None automatically.")
    
    TURN_ANGLE_THRESH = 45 #degrees 
    ANG_VEL_THRESH = 5 #deg/s

    ang_vel, timestamps = get_nth_yaw_deriv(track_uuid, 1, log_dir, coordinate_frame='self', in_degrees=True)
    turn_dict = {'left': [], 'right':[]}

    start_index = 0
    end_index = start_index
    
    while start_index < len(timestamps)-1:
        #Check if the object is continuing to turn in the same direction
        if ((ang_vel[start_index] > 0 and ang_vel[end_index] > 0 
        or ang_vel[start_index] < 0 and ang_vel[end_index] < 0) 
        and end_index < len(timestamps)-1):
            end_index += 1
        else:
            #Check if the object's angle has changed enough to define a turn
            s_per_timestamp = 1/(timestamps[1] - timestamps[0] / 1E9)
            if np.sum(ang_vel[start_index:end_index+1]*s_per_timestamp) > TURN_ANGLE_THRESH:
                turn_dict['left'].extend(timestamps[start_index:end_index+1])
            elif np.sum(ang_vel[start_index:end_index+1]*s_per_timestamp) < -TURN_ANGLE_THRESH:
                turn_dict['right'].extend(timestamps[start_index:end_index+1])
            elif (unwrap_func(near_intersection)(track_uuid, log_dir) 
            and (start_index == 0 and unwrap_func(near_intersection)(track_uuid, log_dir)[0] == timestamps[0]
                or end_index == len(timestamps)-1 and unwrap_func(near_intersection)(track_uuid, log_dir)[-1] == timestamps[-1])):

                if (((start_index==0 and ang_vel[start_index] > ANG_VEL_THRESH) 
                    or (end_index==len(timestamps)-1 and ang_vel[end_index] > ANG_VEL_THRESH))
                and np.mean(ang_vel[start_index:end_index]) > ANG_VEL_THRESH):
                    turn_dict['left'].extend(timestamps[start_index:end_index+1])
                elif (((start_index==0 and ang_vel[start_index] < -ANG_VEL_THRESH) 
                    or (end_index==len(timestamps)-1 and ang_vel[end_index] < -ANG_VEL_THRESH))
                and np.mean(ang_vel[start_index:end_index]) < -ANG_VEL_THRESH):
                    turn_dict['right'].extend(timestamps[start_index:end_index+1])

            start_index = end_index
            end_index += 1 
    
    if direction:
        return turn_dict[direction]
    else:
        return turn_dict['left'] + turn_dict['right']    

@composable
def changing_lanes(
    track_candidates:Union[list, dict], 
    log_dir:Path,
    direction:Literal["left", "right", None]=None) -> dict:
    """
    Identifies lane change events for tracked objects in a scenario.

    Args:
        track_candidates: The tracks to analyze (list of UUIDs or scenario dictionary).
        log_dir: Path to scenario logs.
        direction: The direction of the lane change. None indicates tracking either left or right lane changes ('left', 'right', None).

    Returns:
        dict: 
            A filtered scenario dictionary where:
            Keys are track UUIDs that meet the lane change criteria.
            Values are nested dictionaries containing timestamps and related data.

    Example:
        left_lane_changes = changing_lanes(vehicles, log_dir, direction='left')
    """
    track_uuid = track_candidates

    if direction is not None and direction != 'right' and direction != 'left':
        print("Direction must be 'right', 'left', or None.")
        print("Setting direction to None.")
        direction = None

    lane_traj = get_scenario_lanes(track_uuid, log_dir)
    positions, timestamps = get_nth_pos_deriv(track_uuid, 0, log_dir)
    #Each index stored in dict indicates the exact timestep where the track crossed lanes
    lane_changes_exact = {'left': [], 'right':[]}
    for i in range(1, len(timestamps)):
        prev_lane = lane_traj[timestamps[i-1]]
        cur_lane = lane_traj[timestamps[i]]

        if prev_lane and cur_lane:
            if prev_lane.right_neighbor_id == cur_lane.id:
                lane_changes_exact['right'].append(i)
            elif prev_lane.left_neighbor_id == cur_lane.id:
                lane_changes_exact['left'].append(i)

    lane_changes = {'left': [], 'right':[]}

    for index in lane_changes_exact['left']:
        lane_change_start = index - 1
        lane_change_end = index

        while lane_change_start > 0:
            _, pos_along_width0 = get_pos_within_lane(positions[lane_change_start], lane_traj[timestamps[lane_change_start]])
            _, pos_along_width1 = get_pos_within_lane(positions[lane_change_start+1], lane_traj[timestamps[lane_change_start+1]])

            if (pos_along_width0 and pos_along_width1 and pos_along_width0 > pos_along_width1) or lane_change_start == index-1:
                lane_changes['left'].append(timestamps[lane_change_start])
                lane_change_start -= 1
            else:
                break
            
        while lane_change_end < len(timestamps):
            _, pos_along_width0 = get_pos_within_lane(positions[lane_change_end-1], lane_traj[timestamps[lane_change_end-1]])
            _, pos_along_width1 = get_pos_within_lane(positions[lane_change_end], lane_traj[timestamps[lane_change_end]])

            if (pos_along_width0 and pos_along_width1 and pos_along_width0 > pos_along_width1) or lane_change_end == index:
                lane_changes['left'].append(timestamps[lane_change_end])
                lane_change_end += 1
            else:
                break
    
    for index in lane_changes_exact['right']:
        lane_change_start = index - 1
        lane_change_end = index

        while lane_change_start > 0:
            _, pos_along_width0 = get_pos_within_lane(positions[lane_change_start], lane_traj[timestamps[lane_change_start]])
            _, pos_along_width1 = get_pos_within_lane(positions[lane_change_start+1], lane_traj[timestamps[lane_change_start+1]])

            if pos_along_width0 and pos_along_width1 and pos_along_width0 < pos_along_width1 or lane_change_start == index-1:
                lane_changes['right'].append(timestamps[lane_change_start])
                lane_change_start -= 1
            else:
                break
            
        while lane_change_end < len(timestamps):
            _, pos_along_width0 = get_pos_within_lane(positions[lane_change_end-1], lane_traj[timestamps[lane_change_end-1]])
            _, pos_along_width1 = get_pos_within_lane(positions[lane_change_end], lane_traj[timestamps[lane_change_end]])

            if pos_along_width0 and pos_along_width1 and pos_along_width0 < pos_along_width1 or lane_change_end == index:
                lane_changes['right'].append(timestamps[lane_change_end])
                lane_change_end += 1
            else:
                break

    if direction:
        lane_changing_timestamps = lane_changes[direction]
    else:
        lane_changing_timestamps = sorted(list(set(lane_changes['left'] + (lane_changes['right']))))

    turning_timestamps = unwrap_func(turning)(track_uuid, log_dir)
    return sorted(list(set(lane_changing_timestamps).difference(set(turning_timestamps))))
    
@composable
def has_lateral_acceleration(
    track_candidates:Union[list,dict],
    log_dir:Path,
    min_accel=-np.inf,
    max_accel=np.inf) -> dict:
    """
    Objects with a lateral acceleartion between the minimum and maximum thresholds. 
    Most objects with a high lateral acceleration are turning. Postive values indicate accelaration
    to the left while negative values indicate acceleration to the right. 

    Args:
        track_candidates: The tracks to analyze (list of UUIDs or scenario dictionary).
        log_dir: Path to scenario logs.
        direction: The direction of the lane change. None indicates tracking either left or right lane changes ('left', 'right', None).

    Returns:
        dict: 
            A filtered scenario dictionary where:
            Keys are track UUIDs that meet the lane change criteria.
            Values are nested dictionaries containing timestamps and related data.

    Example:
        jerking_left = has_lateral_acceleration(non_turning_vehicles, log_dir, min_accel=2)
    """
    track_uuid = track_candidates

    hla_timestamps = []
    accelerations, timestamps = get_nth_pos_deriv(track_uuid, 2, log_dir, coordinate_frame='self')
    for i, accel in enumerate(accelerations):
        if min_accel <= accel[1] <= max_accel: #m/s^2
            hla_timestamps.append(timestamps[i])
    return hla_timestamps
        

def unwrap_func(decorated_func: Callable, n=1) -> Callable:
    """Get the original function from a decorated function."""

    unwrapped_func = decorated_func
    for _ in range(n):
        if hasattr(unwrapped_func, '__wrapped__'):
            unwrapped_func = unwrapped_func.__wrapped__
        else:
            break
    
    return unwrapped_func

def parallelize_uuids(
    func: Callable,
    all_uuids: list[str],
    *args,
    **kwargs
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Parallelize UUID processing using Pathos ProcessingPool.
    
    Notes:
        - Pathos provides better serialization than standard multiprocessing
        - ProcessingPool.map() is already synchronous and will wait for completion
        - Pathos handles class methods and nested functions better than multiprocessing
    """
    func = unwrap_func(func)

    def worker_func(uuid: str) -> tuple[str, Any, Any]:
        """
        Worker function wrapper that maintains closure over func and its arguments.
        Pathos handles this closure better than standard multiprocessing.
        """
        result = func(uuid, *args, **kwargs)
        if not isinstance(result, tuple):
            result = (result, None)
        timestamps = result[0]
        related = result[1]
        
        return uuid, timestamps, related

    # Initialize the pool
    print(cache_manager.num_processes)
    with Pool(nodes=cache_manager.num_processes) as pool:
        # Map work to the pool - this will wait for completion
        results = pool.map(worker_func, all_uuids)
    
    # Process results
    uuid_dict = {}
    related_dict = {}
    
    for uuid, timestamps, related in results: 
        if timestamps is not None:
            uuid_dict[uuid] = timestamps
            related_dict[uuid] = related
    
    return uuid_dict, related_dict
        

def is_point_in_polygon(point, polygon):
    """
    Determine if a point is inside a polygon using the ray-casting algorithm.

    :param point: (x, y) coordinates of the point.
    :param polygon: List of (x, y) coordinates defining the polygon vertices.
    :return: True if the point is inside the polygon, False otherwise.
    """
    x, y = point
    n = len(polygon)
    inside = False

    px1, py1 = polygon[0]
    for i in range(1, n + 1):
        px2, py2 = polygon[i % n]
        if y > min(py1, py2):
            if y <= max(py1, py2):
                if x <= max(px1, px2):
                    if py1 != py2:
                        xinters = (y - py1) * (px2 - px1) / (py2 - py1) + px1
                    if px1 == px2 or x <= xinters:
                        inside = not inside
        px1, py1 = px2, py2

    return inside


def polygons_overlap(poly1, poly2):
    """
    Determine if two polygons overlap using the Separating Axis Theorem (SAT).
    
    Parameters:
    poly1, poly2: Nx2 numpy arrays where each row is a vertex (x,y)
                 First and last vertices should be the same
    visualize: bool, whether to show a visualization of the polygons
    
    Returns:
    bool: True if polygons overlap, False otherwise
    """
    def get_edges(polygon):
        # Get all edges of the polygon as vectors
        return [polygon[i+1] - polygon[i] for i in range(len(polygon)-1)]
    
    def get_normal(edge):
        # Get the normal vector to an edge
        return np.array([-edge[1], edge[0]])
    
    def project_polygon(polygon, axis):
        # Project all vertices onto an axis
        dots = [np.dot(vertex, axis) for vertex in polygon]
        return min(dots), max(dots)
    
    def overlap_on_axis(min1, max1, min2, max2):
        # Check if projections overlap
        return (min1 <= max2 and min2 <= max1) \
                or (min1<=min2 and max1>=max2) \
                or (min2<=min1 and max2>=max1)
    
    # Get all edges from both polygons
    edges1 = get_edges(poly1)
    edges2 = get_edges(poly2)
    
    # Test all normal vectors as potential separating axes
    for edge in edges1 + edges2:
        # Get the normal to the edge
        normal = get_normal(edge)
        
        # Normalize the normal vector
        normal = normal / np.linalg.norm(normal)
        
        # Project both polygons onto the normal
        min1, max1 = project_polygon(poly1, normal)
        min2, max2 = project_polygon(poly2, normal)
        
        # If we find a separating axis, the polygons don't overlap
        if not overlap_on_axis(min1, max1, min2, max2):
            return False
    
    # If we get here, no separating axis was found, so the polygons overlap
    return True


def plot_cuboids(cuboids: list[Cuboid], plotter: pv.Plotter, transforms: list[SE3], color='red', opacity=.25, with_front = False,
    with_cf = False, with_label=False) -> list[vtk.vtkActor]:
    """
    Plot a cuboid using its vertices from the Cuboid class pattern
    
    vertices pattern from the class:
        5------4
        |\\    |\\
        | \\   | \\
        6--\\--7  \\
        \\  \\  \\ \\
    l    \\  1-------0    h
     e    \\ ||   \\ ||   e
      n    \\||    \\||   i
       g    \\2------3    g
        t      width      h
         h                t
    """
    actors = []
    combined_mesh = None
    combined_front = None
    combined_cfs = None

    if not cuboids:
        return actors
    
    if isinstance(transforms, SE3):
        transforms = [transforms] * len(cuboids)

    for i, cuboid in enumerate(cuboids):
        #Ego vehicle use ego coordintate frame (centered on rear axel).
        #All other objects have a coordinate frame centered on their centroid.
        if cuboid.category == 'EGO_VEHICLE':
            ego_vertices = cuboid.vertices_m + 0
            #Adjusting bounding box to be centered on centroid of 2018 Ford Fusion Hybrid
            ego_vertices[:,0] += 1.422
            ego_vertices[:,2] += 0.25
            vertices = transforms[i].transform_from(ego_vertices)
        else:
            vertices = transforms[i].transform_from(cuboid.vertices_m)

        if with_front:
            front_face = [4,0, 1, 2, 3],  # front

            if not combined_front:
                combined_front = pv.PolyData(vertices, front_face)
            else:
                front_surface = pv.PolyData(vertices, front_face)
                combined_front = combined_front.append_polydata(front_surface)

        # Create faces using the vertex indices
        faces = [
            [4,4, 5, 6, 7],     # back
            [4,0, 4, 7, 3],     # right
            [4,1, 5, 6, 2],     # left
            [4,0, 1, 5, 4],     # top
            [4,2, 3, 7, 6]      # bottom
        ]
        
        # Create a PolyData object for the cuboid
        if not combined_mesh:
            combined_mesh = pv.PolyData(vertices, faces)
        else:
            surface = pv.PolyData(vertices, faces)
            combined_mesh = combined_mesh.append_polydata(surface)

        if with_label:
            category = cuboid.category
            center = vertices.mean(axis=0)
            # Create a point at the center
            point = pv.PolyData(center)
            # Add the category as a label
            labels = plotter.add_point_labels(
                point, 
                [str(category)], 
                point_size=.1,  # Make the point invisible
                font_size=10,
                show_points=False,
                shape_opacity=0.3,     # Semi-transparent background
                font_family='arial'
            )
            actors.append(labels)
        
        if with_cf:
            combined_cfs = append_cf_mesh(combined_cfs, cuboid, transforms[i])


    all_cuboids_actor = plotter.add_mesh(combined_mesh, color=color, opacity=opacity, pickable=False, lighting=False)
    actors.append(all_cuboids_actor)

    if with_cf:
        all_cfs_actor = plotter.add_mesh(combined_cfs, color='black', line_width=3,opacity=opacity, pickable=False, lighting=False)
        actors.append(all_cfs_actor)

    if with_front:
        all_fronts_actor = plotter.add_mesh(combined_front, color='yellow', opacity=opacity, pickable=False, lighting=False)
        actors.append(all_fronts_actor)
    
    return actors


def append_cf_mesh(combined_cfs:pv.PolyData, cuboid:Cuboid, transform:SE3=None):

    x_line = np.array([[0,0,0],[10,0,0]])
    y_line = np.array([[0,0,0],[0,5,0]])

    x_line = transform.compose(cuboid.dst_SE3_object).transform_from(x_line)
    y_line = transform.compose(cuboid.dst_SE3_object).transform_from(y_line)

    pv_xline = pv.PolyData(x_line)
    pv_xline.lines = np.array([2,0,1])

    pv_yline = pv.PolyData(y_line)
    pv_yline.lines = np.array([2,0,1])

    if combined_cfs is None:
        combined_cfs = pv_xline
        combined_cfs = combined_cfs.append_polydata(pv_yline)
    else:
        combined_cfs = combined_cfs.append_polydata(pv_xline)
        combined_cfs = combined_cfs.append_polydata(pv_yline)

    return combined_cfs


@cache_manager.create_cache('get_nth_pos_deriv')
def get_nth_pos_deriv(
    track_uuid, 
    n, 
    log_dir, 
    coordinate_frame=None,
    direction='forward') -> tuple[np.ndarray, np.ndarray]:

    """Returns the nth positional derivative of the track at all timestamps 
    with respect to city coordinates. """

    df = read_feather(log_dir / 'sm_annotations.feather')
    ego_poses = get_ego_SE3(log_dir)

    # Filter the DataFrame
    cuboid_df = df[df['track_uuid'] == track_uuid]
    ego_coords = cuboid_df[['tx_m', 'ty_m', 'tz_m']].to_numpy()

    timestamps = cuboid_df['timestamp_ns'].to_numpy()
    city_coords = np.zeros((ego_coords.shape)).T
    for i in range(len(ego_coords)):
        city_coords[:,i] = ego_poses[timestamps[i]].transform_from(ego_coords[i,:])

    city_coords = city_coords.T

    #Very often, different cuboids are not seen by the ego vehicle at the same time.
    #Only the timestamps where both cuboids are observed are calculated.
    if type(coordinate_frame) == str and coordinate_frame != get_ego_uuid(log_dir):
        if coordinate_frame == 'self':
            coordinate_frame = track_uuid

        cf_df = df[df['track_uuid'] == coordinate_frame]
        cf_timestamps = cf_df['timestamp_ns'].to_numpy()

        new_timestamps = np.array(list(set(cf_timestamps).intersection(set(timestamps))))
        new_timestamps.sort(axis=0)

        city_coords = city_coords[np.isin(timestamps, new_timestamps)]
        timestamps = new_timestamps
        cf_df = cf_df[cf_df['timestamp_ns'].isin(timestamps)]
    
    INTERPOLATION_RATE = 1
    prev_deriv = np.copy(city_coords)
    next_deriv = np.zeros(prev_deriv.shape)
    for _ in range(n):
        next_deriv=np.zeros(prev_deriv.shape)
        if len(timestamps) == 1:
            break

        for i in range(len(prev_deriv)):
            past_index = max(0, i-INTERPOLATION_RATE)
            future_index = min(len(timestamps)-1, i+INTERPOLATION_RATE)

            next_deriv[i] = 1e9*(prev_deriv[future_index]-prev_deriv[past_index])/(timestamps[future_index]-timestamps[past_index])

        prev_deriv=np.copy(next_deriv)
    
    if len(timestamps) <= 1:
        pos_deriv = prev_deriv
    else:
        pos_deriv = scipy.ndimage.median_filter(prev_deriv, size=min(7,len(prev_deriv)), mode='nearest', axes=0) 

    if coordinate_frame == get_ego_uuid(log_dir):
        for i in range(len(pos_deriv)):
            city_to_ego = ego_poses[timestamps[i]].inverse()
            pos_deriv[i] = city_to_ego.transform_from(pos_deriv[i])
            if n != 0:
                #Velocity/acceleration/jerk vectors only need to be rotated
                pos_deriv[i] -= city_to_ego.translation
    elif type(coordinate_frame) == str:

        cf_df = df[df['track_uuid'] == coordinate_frame]
        if cf_df.empty:
            print('Coordinate frame must be None, \'ego\', \'self\', track_uuid, or city to coordinate frame SE3 object.')
            print('Returning answer in city coordinates')
            return pos_deriv, timestamps
        
        cf_df = cf_df[cf_df['timestamp_ns'].isin(timestamps)]
        cf_list = CuboidList.from_dataframe(cf_df)

        for i in range(len(pos_deriv)):
            city_to_ego = ego_poses[timestamps[i]].inverse()  
            ego_to_self = cf_list[i].dst_SE3_object.inverse() 
            city_to_self = ego_to_self.compose(city_to_ego)   
            pos_deriv[i] = city_to_self.transform_from(pos_deriv[i])
            if n != 0:
                #Velocity/acceleration/jerk vectors only need to be rotated
                pos_deriv[i] -= city_to_self.translation
    elif type(coordinate_frame) == SE3:
        pos_deriv = (coordinate_frame.transform_from(pos_deriv.T)).T
    #elif type(coordinate_frame) == list and coordinate_frame and type(coordinate_frame[0]) == SE3 and len(coordinate_frame) == len(prev_deriv):
    elif coordinate_frame is not None:
        print('Coordinate frame must be None, \'ego\', \'self\', track_uuid, or city to coordinate frame SE3 object.')

    if direction == 'left':
        rot_mat = np.array([[0,1,0],[-1,0,0],[0,0,1]])
    elif direction == 'right':
        rot_mat = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    elif direction == 'backward':
        rot_mat = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
    else:
        rot_mat = np.eye(3)

    pos_deriv = (rot_mat @ pos_deriv.T).T

    return pos_deriv, timestamps

def get_nth_radial_deriv(track_uuid, n, log_dir, 
    coordinate_frame=None)->tuple[np.ndarray, np.ndarray]:

    relative_pos, timestamps = get_nth_pos_deriv(track_uuid, 0, log_dir, coordinate_frame=coordinate_frame)
    
    distance = np.linalg.norm(relative_pos, axis=1)
    radial_deriv = distance
    for i in range(n):
        if len(radial_deriv) > 1:
            radial_deriv = np.gradient(radial_deriv)
        else:
            radial_deriv = np.array([0])

    return radial_deriv, timestamps

@composable_relational
def facing_toward(
    track_candidates:Union[list,dict],
    related_candidates:Union[list,dict],
    log_dir:Path,
    within_angle:float=22.5,
    max_distance:float=np.inf)->dict:
    """
    Identifies objects in track_candidates that are facing toward objects in related candidates.
    The related candidate must lie within a region lying within within_angle degrees on either side the track-candidate's forward axis.

    Args:
        track_candidates: The tracks that could be heading toward another tracks
        related_candidates: The objects to analyze to see if the track_candidates are heading toward
        log_dir:  Path to the directory containing scenario logs and data.
        fov: The field of view of the track_candidates. The related candidate must lie within a region lying 
            within fov/2 degrees on either side the track-candidate's forward axis.
        max_distance: The maximum distance a related_candidate can be away to be considered by 

    Returns:
        A filtered scenario dict that contains the subset of track candidates heading toward at least one of the related candidates.

    Example:
        pedestrian_facing_away = scenario_not(facing_toward)(pedestrian, ego_vehicle, log_dir, within_angle=180)
    """

    track_uuid = track_candidates
    facing_toward_timestamps = []
    facing_toward_objects = {}

    for candidate_uuid in related_candidates:
        if candidate_uuid == track_uuid:
            continue

        traj, timestamps = get_nth_pos_deriv(candidate_uuid, 0, log_dir, coordinate_frame=track_uuid)
        for i, timestamp in enumerate(timestamps):

            angle = np.rad2deg(np.arctan2(traj[i, 1],  traj[i,0]))
            dist = np.linalg.norm(traj[i])

            if np.abs(angle) <= within_angle and dist <= max_distance:
                facing_toward_timestamps.append(timestamp)

                if candidate_uuid not in facing_toward_objects:
                    facing_toward_objects[candidate_uuid] = []
                facing_toward_objects[candidate_uuid].append(timestamp)

    return facing_toward_timestamps, facing_toward_objects
    
@composable_relational
def heading_toward(
    track_candidates:Union[list,dict],
    related_candidates:Union[list,dict],
    log_dir:Path,
    angle_threshold:float=22.5,
    minimum_speed:float=.5,
    max_distance:float=np.inf)->dict:
    """
    Identifies objects in track_candidates that are heading toward objects in related candidates.
    The track candidates acceleartion vector must be within the given angle threshold of the relative position vector.
    The track candidates must have a component of velocity toward the related candidate greater than the minimum_accel.

    Args:
        track_candidates: The tracks that could be heading toward another tracks
        related_candidates: The objects to analyze to see if the track_candidates are heading toward
        log_dir:  Path to the directory containing scenario logs and data.
        angle_threshold: The maximum angular difference between the velocity vector and relative position vector between
            the track candidate and related candidate.
        min_vel: The minimum magnitude of the component of velocity toward the related candidate
        max_distance: Distance in meters the related candidates can be away from the track candidate to be considered

    Returns:
        A filted scenario dict that contains the subset of track candidates heading toward at least one of the related candidates.


    Example:
        heading_toward_traffic_cone = heading_toward(vehicles, traffic_cone, log_dir)
    """

    track_uuid = track_candidates
    heading_toward_timestamps = []
    heading_toward_objects = {}

    track_vel, track_timestamps = get_nth_pos_deriv(track_uuid, 1, log_dir, coordinate_frame=track_uuid)

    for candidate_uuid in related_candidates:
        if candidate_uuid == track_uuid:
            continue

        related_pos, _ = get_nth_pos_deriv(candidate_uuid, 0, log_dir, coordinate_frame=track_uuid)
        track_radial_vel, related_timestamps = get_nth_radial_deriv(
            track_uuid, 1, log_dir, coordinate_frame=candidate_uuid)
        
        for i, timestamp in enumerate(related_timestamps):
            timestamp_vel = track_vel[np.where(track_timestamps == timestamp)]
            vel_direction = timestamp_vel/np.linalg.norm(timestamp_vel)
            direction_of_related = related_pos[i]/np.linalg.norm(related_pos[i])
            angle = np.rad2deg(np.arccos(np.dot(vel_direction, direction_of_related)))

            if -track_radial_vel[i] >= minimum_speed and angle <= angle_threshold \
            and np.linalg.norm(related_pos[i]) <= max_distance:

                heading_toward_timestamps.append(timestamp)
                if candidate_uuid not in heading_toward_objects:
                    heading_toward_objects[candidate_uuid] = []
                heading_toward_objects[candidate_uuid].append(timestamp)

    return heading_toward_timestamps, heading_toward_objects

@composable_relational
def accelerating_toward(
    track_candidates:Union[list,dict],
    related_candidates:Union[list,dict],
    log_dir:Path,
    angle_threshold:float=22.5,
    min_accel:float=1,
    max_distance:float=30)->dict:
    """
    Identifies objects in track_candidates that are accelerating toward objects in related candidates.
    The track candidates acceleartion vector must be within the given angle threshold of the relative position vector.
    The track candidates must have a component of acceleration toward the related candidate greater than the minimum_accel.

    Args:
        track_candidates: The tracks that could be accelerating toward another tracks
        related_candidates: The objects to analyze to see if the track_candidates are accelerating toward
        log_dir:  Path to the directory containing scenario logs and data.
        angle_threshold: The maximum angular difference in degrees between the acceleration vector and relative position vector between
            the track candidate and related candidate.
        min_accel: The minimum magnitude of the component of acceleration toward the related candidate
        max_distance: Distance in meters the related candidates can be away from the track candidate to be considered

    Returns:
        A filted scenario dict that contains the subset of track candidates accelerating toward at least one of the related candidates.

    Example:
        accelerating_toward_pedestrian = accelerating_toward(vehicles, pedestrians, log_dir)
    """

    track_uuid = track_candidates
    accel_toward_timestamps = []
    accel_toward_objects = {}

    track_accel, track_timestamps = get_nth_pos_deriv(track_uuid, 2, log_dir, coordinate_frame=track_uuid)

    for candidate_uuid in related_candidates:
        if candidate_uuid == track_uuid:
            continue

        relative_pos, _ = get_nth_pos_deriv(candidate_uuid, 0, log_dir, coordinate_frame=track_uuid)
        track_radial_accel, related_timestamps = get_nth_radial_deriv(
            track_uuid, 2, log_dir, coordinate_frame=candidate_uuid)
        
        for i, timestamp in enumerate(related_timestamps):
            timestamp_accel = track_accel[np.where(track_timestamps == timestamp)]
            accel_direction = timestamp_accel/np.linalg.norm(timestamp_accel)
            relative_direction = relative_pos[i]/np.linalg.norm(relative_pos[i])
            angle = np.rad2deg(np.arccos(np.dot(accel_direction, relative_direction)))

            if -track_radial_accel[i] >= min_accel and angle <= angle_threshold \
            and np.linalg.norm(relative_pos[i]) <= max_distance:

                accel_toward_timestamps.append(timestamp)
                if candidate_uuid not in accel_toward_objects:
                    accel_toward_objects[candidate_uuid] = []
                accel_toward_objects[candidate_uuid].append(timestamp)

    return accel_toward_timestamps, accel_toward_objects

@composable
def accelerating(
    track_candidates:Union[list,dict],
    log_dir:Path,
    min_accel:float=1,
    max_accel:float=np.inf,
    thresh:float=None)->dict:
    """
    Identifies objects in track_candidates that have a forward acceleration above a threshold.
    Values under -1 reliably indicates braking. Values over 1.0 reliably indiciates accelerating.

    Args:
        track_candidates: The tracks to analyze for acceleration (list of UUIDs or scenario dictionary)
        log_dir:  Path to the directory containing scenario logs and data.
        min_accel: The lower bound of acceleration considered
        max_accel: The upper bound of acceleration considered

    Returns:
        A filtered scenario dictionary containing the objects with an acceleration between the lower and upper bounds.

    Example:
        accelerating_motorcycles = accelerating(motorcycles, log_dir)

    """
    track_uuid = track_candidates

    #conversion for scenarios that use legacy accelerating function
    if thresh is not None:
        min_accel = thresh
        max_accel = np.inf

    acc_timestamps = []
    accelerations, timestamps = get_nth_pos_deriv(track_uuid, 2, log_dir, coordinate_frame='self')
    for i, accel in enumerate(accelerations):
        if min_accel <= accel[0] <= max_accel: #m/s^2
            acc_timestamps.append(timestamps[i])
    return acc_timestamps


@composable
def has_velocity(
    track_candidates:Union[list,dict],
    log_dir:Path, 
    min_velocity:float=.5, 
    max_velocity:float=np.inf)->dict:
    """
    Identifies objects with a velocity between the given maximum and minimum velocities in m/s.
    Stationary objects may have a velocity up to 0.5 m/s due to annotation jitter.

    Args:
        track_candidates: Tracks to analyze (list of UUIDs or scenario dictionary).
        log_dir: Path to scenario logs.
        min_velocity: Minimum velocity (m/s). Defaults to 0.5.

    Returns:
        Filtered scenario dictionary of objects meeting the velocity criteria.

    Example:
        fast_vehicles = has_min_velocity(vehicles, log_dir, min_velocity=5)
    """
    track_uuid = track_candidates

    vel_timestamps = []
    vels, timestamps = get_nth_pos_deriv(track_uuid, 1, log_dir)
    for i, vel in enumerate(vels):
        if min_velocity <= np.linalg.norm(vel) <= max_velocity: #m/s
            vel_timestamps.append(timestamps[i])
    return vel_timestamps


#@cache_manager.create_cache('get_nth_yaw_deriv')
def get_nth_yaw_deriv(track_uuid, n, log_dir, coordinate_frame=None, in_degrees=False):
    """Returns the nth angular derivative of the track at all timestamps 
    with respect to the given coordinate frame. The default coordinate frame is city.
    The returned angle is yaw measured from the x-axis of the track coordinate frame to the x-axis
    of the source coordinate frame"""

    df = read_feather(log_dir / 'sm_annotations.feather')
    ego_poses = get_ego_SE3(log_dir)

    # Filter the DataFrame
    cuboid_df = df[df['track_uuid'] == track_uuid]
    cuboid_list = CuboidList.from_dataframe(cuboid_df)

    self_to_ego_list:list[SE3] = []

    for i in range(len(cuboid_list)):
        self_to_ego_list.append(cuboid_list[i].dst_SE3_object)

    timestamps = cuboid_df['timestamp_ns'].to_numpy()
    self_to_city_list = []
    for i in range(len(self_to_ego_list)):
        self_to_city_list.append(ego_poses[timestamps[i]].compose(self_to_ego_list[i]))

    #Very often, different cuboids are not seen by the ego vehicle at the same time.
    #Only the timestamps where both cuboids are observed are calculated.
    if type(coordinate_frame) == str and coordinate_frame != get_ego_uuid(log_dir):
        if coordinate_frame == 'self':
            coordinate_frame = track_uuid

        cf_df = df[df['track_uuid'] == coordinate_frame]
        cf_timestamps = cf_df['timestamp_ns'].to_numpy()

        if cf_df.empty:
            print('Coordinate frame must be None, \'ego\', \'self\', track_uuid, or city to coordinate frame SE3 object.')
            print('Returning answer in city coordinates')
        else:
            new_timestamps = np.array(list(set(cf_timestamps).intersection(set(timestamps))))
            new_timestamps.sort(axis=0)

            filtered_timestamps = np.isin(timestamps, new_timestamps)

            # Convert mask to indices
            filtered_indices = np.where(filtered_timestamps)[0]

            # Index the list
            filtered_list = [self_to_city_list[i] for i in filtered_indices]
            self_to_city_list = filtered_list
            timestamps = new_timestamps

    city_yaws = np.zeros((len(self_to_city_list),3))
    for i in range(len(self_to_city_list)):
        city_yaws[i] = Rotation.from_matrix(self_to_city_list[i].rotation).as_rotvec()

    INTERPOLATION_RATE = 1
    prev_deriv = np.copy(city_yaws)
    next_deriv = np.zeros(prev_deriv.shape)
    for j in range(n):
        next_deriv=np.zeros(prev_deriv.shape)
        if len(timestamps) == 1:
            break
    
        for i in range(len(prev_deriv)):
            past_index = max(i-INTERPOLATION_RATE,0)
            future_index = min(i+INTERPOLATION_RATE, len(prev_deriv)-1)
            
            difference = prev_deriv[future_index] - prev_deriv[past_index]
            for k in range(len(prev_deriv[0])):
                if j == 0 and abs(difference[k]) > np.pi:
                        if difference[k] > 0:
                            difference[k] -= 2*np.pi
                        else:
                            difference[k] += 2*np.pi

            next_deriv[i] = 1e9*difference/(timestamps[future_index]-timestamps[past_index])

        prev_deriv=np.copy(next_deriv)

    cf_angles = np.copy(prev_deriv)

    if n == 0 and coordinate_frame == get_ego_uuid(log_dir):
        for i in range(len(prev_deriv)):
            city_to_ego = ego_poses[timestamps[i]].inverse().rotation
            cf_angles[i] = Rotation.from_matrix(city_to_ego @ Rotation.from_rotvec(prev_deriv[i]).as_matrix()).as_rotvec()
    elif n == 0 and type(coordinate_frame) == str:
        cf_df = df[df['track_uuid'] == coordinate_frame]
        if not cf_df.empty:
            cf_list = CuboidList.from_dataframe(cf_df)
            for i in range(len(prev_deriv)):
                city_to_ego = ego_poses[timestamps[i]].inverse()
                ego_to_obj = cf_list[i].dst_SE3_object.inverse()
                city_to_obj = ego_to_obj.compose(city_to_ego).rotation
                cf_angles[i] = Rotation.from_matrix(city_to_obj @ Rotation.from_rotvec(prev_deriv[i]).as_matrix()).as_rotvec()
    elif n == 0 and type(coordinate_frame) == SE3:
        for i in range(len(prev_deriv)):
            cf_angles[i] = Rotation.from_matrix(coordinate_frame.rotation @ Rotation.from_rotvec(prev_deriv[i]).as_matrix()).as_rotvec()
    elif n==0 and coordinate_frame is not None:
        print('Coordinate frame must be None, \'ego\', \'self\', track_uuid, or city to coordinate frame SE3 object.')

    if in_degrees: 
        cf_angles = np.rad2deg(cf_angles)

    return cf_angles[:,2], timestamps


@composable
def at_pedestrian_crossing(
    track_candidates:Union[list,dict],
    log_dir:Path,
    within_distance:float=1)->dict:
    """
    Identifies objects that within a certain distance from a pedestrian crossing. A distance of zero indicates
    that the object is within the boundaries of the pedestrian crossing.

    Args:
        track_candidates: Tracks to analyze (list of UUIDs or scenario dictionary).
        log_dir: Path to scenario logs.
        within_distance: Distance in meters the track candidate must be from the pedestrian crossing. A distance of zero
            means that the object must be within the boundaries of the pedestrian crossing.

    Returns:
        Filtered scenario dictionary where keys are track UUIDs and values are lists of timestamps.

    Example:
        vehicles_at_ped_crossing = at_pedestrian_crossing(vehicles, log_dir)
    """
    track_uuid = track_candidates

    avm = get_map(log_dir)
    ped_crossings = avm.get_scenario_ped_crossings()

    timestamps = get_timestamps(track_uuid, log_dir)
    ego_poses = get_ego_SE3(log_dir)

    timestamps_at_object = []
    for timestamp in timestamps:
        track_cuboid = get_cuboid_from_uuid(track_uuid, log_dir, timestamp=timestamp)
        city_vertices = ego_poses[timestamp].transform_from(track_cuboid.vertices_m) 
        track_poly = np.array([city_vertices[2],city_vertices[6],city_vertices[7],city_vertices[3],city_vertices[2]])[:,:2]

        for ped_crossing in ped_crossings:
            pc_poly = ped_crossing.polygon
            pc_poly = dilate_convex_polygon(pc_poly[:,:2], distance=within_distance)
            ped_crossings = get_pedestrian_crossings(avm, track_poly)

            if polygons_overlap(track_poly, pc_poly):
                timestamps_at_object.append(timestamp)
        
    return timestamps_at_object


@composable
def on_lane_type(
    track_uuid:Union[list,dict],
    log_dir,
    lane_type:Literal["BUS", "VEHICLE", "BIKE"])->dict:
    """
    Identifies objects on a specific lane type.

    Args:
        track_candidates: Tracks to analyze (list of UUIDs or scenario dictionary).
        log_dir: Path to scenario logs.
        lane_type: Type of lane to check ('BUS', 'VEHICLE', or 'BIKE').

    Returns:
        Filtered scenario dictionary where keys are track UUIDs and values are lists of timestamps.

    Example:
        vehicles_on_bus_lane = on_lane_type(vehicles, log_dir, lane_type="BUS")
    """

    traj, timestamps = get_nth_pos_deriv(track_uuid, 0, log_dir)
    scenario_lanes = get_scenario_lanes(track_uuid, log_dir, traj=traj, timestamps=timestamps)

    return [timestamp for timestamp in timestamps if scenario_lanes[timestamp] and scenario_lanes[timestamp].lane_type == lane_type]


@composable
def near_intersection(
    track_uuid:Union[list,dict],
    log_dir:Path,
    threshold:float=5)->dict:
    """
    Identifies objects within a specified threshold of an intersection in meters.

    Args:
        track_candidates: Tracks to analyze (list of UUIDs or scenario dictionary).
        log_dir: Path to scenario logs.
        threshold: Distance threshold (in meters) to define "near" an intersection.

    Returns:
        Filtered scenario dictionary where keys are track UUIDs and values are lists of timestamps.

    Example:
        bicycles_near_intersection = near_intersection(bicycles, log_dir, threshold=10.0)
    """


    traj, timestamps = get_nth_pos_deriv(track_uuid, 0, log_dir)

    avm = get_map(log_dir)
    lane_segments = avm.get_scenario_lane_segments()

    ls_polys = []
    for ls in lane_segments:
        if ls.is_intersection:
            ls_polys.append(ls.polygon_boundary)

    dilated_intersections = []
    for ls in ls_polys:
        dilated_intersections.append(dilate_convex_polygon(ls[:,:2], threshold))
    
    near_intersection_timestamps = []
    for i, pos in enumerate(traj):
        for dilated_intersection in dilated_intersections:
            if is_point_in_polygon(pos[:2], dilated_intersection):
                near_intersection_timestamps.append(timestamps[i])

    return near_intersection_timestamps

@composable
def on_intersection(track_candidates:Union[list,dict], log_dir:Path):
    """
    Identifies objects located on top of an road intersection.

    Args:
        track_candidates: Tracks to analyze (list of UUIDs or scenario dictionary).
        log_dir: Path to scenario logs.

    Returns:
        Filtered scenario dictionary where keys are track UUIDs and values are lists of timestamps.

    Example:
        strollers_on_intersection = on_intersection(strollers, log_dir)
    """
    track_uuid = track_candidates

    traj, timestamps = get_nth_pos_deriv(track_uuid, 0, log_dir)
    scenario_lanes = get_scenario_lanes(track_uuid, log_dir, traj=traj, timestamps=timestamps)

    timestamps_on_intersection = []
    for timestamp in timestamps:
        if scenario_lanes[timestamp] is not None and scenario_lanes[timestamp].is_intersection:
            timestamps_on_intersection.append(timestamp)

    return timestamps_on_intersection

@cache_manager.create_cache('get_map')
def get_map(log_dir: Path):

    try:
        avm = ArgoverseStaticMap.from_map_dir(log_dir / 'map', build_raster=True)
    except:
        split = get_log_split(log_dir)
        avm = ArgoverseStaticMap.from_map_dir(AV2_DATA_DIR / split / log_dir.name / 'map', build_raster=True)
        
    return avm

def get_log_split(log_dir: Path):

    if log_dir.name in TEST:
        split = 'test'
    elif log_dir.name in TRAIN:
        split = 'train'
    elif log_dir.name in VAL:
        split = 'val'

    return split

def get_ego_SE3(log_dir:Path):
    """Returns list of ego_to_city SE3 transformation matrices"""
    try:
        ego_poses = read_city_SE3_ego(log_dir)
    except:
        split = get_log_split(log_dir)
        ego_poses = read_city_SE3_ego(AV2_DATA_DIR / split / log_dir.name)

    return ego_poses


def dilate_convex_polygon(points, distance):
    """
    Dilates the perimeter of a convex polygon specified in clockwise order by a given distance.
    
    Args:
        points (numpy.ndarray): Nx2 array of (x, y) coordinates representing the vertices of the convex polygon
                                in clockwise order. The first and last points are identical.
        distance (float): Distance to dilate the polygon perimeter. Positive for outward, negative for inward.

    Returns:
        numpy.ndarray: Nx2 array of (x, y) coordinates representing the dilated polygon vertices.
                       The first and last points will also be identical.
    """
    def normalize(v):
        """Normalize a vector."""
        norm = np.linalg.norm(v)
        return v / norm if norm != 0 else v

    n = len(points)  # Account for duplicate closing point
    dilated_points = []

    for i in range(1,n):
        # Current, previous, and next points
        prev_point = points[i - 1]  # Previous vertex (wrap around for first vertex)
        curr_point = points[i]     # Current vertex
        next_point = points[(i + 1) % (n-1)]  # Next vertex (wrap around for last vertex)

        # Edge vectors
        edge1 = normalize(curr_point - prev_point)  # Edge vector from prev to curr
        edge2 = normalize(next_point - curr_point)  # Edge vector from curr to next

        # Perpendicular vectors to edges (flipped for clockwise order)
        perp1 = np.array([edge1[1], -edge1[0]])  # Rotate -90 degrees
        perp2 = np.array([edge2[1], -edge2[0]])  # Rotate -90 degrees

        # Average of perpendiculars (to find outward bisector direction)
        bisector = normalize(perp1 + perp2)

        # Avoid division by zero or near-zero cases
        dot_product = np.dot(bisector, perp1)
        if abs(dot_product) < 1e-10:  # Small threshold for numerical stability
            displacement = distance * bisector  # Fallback: scale bisector direction
        else:
            displacement = distance / dot_product * bisector

        # Compute the new vertex
        new_point = curr_point + displacement
        dilated_points.append(new_point)

    # Add the first point to the end to close the polygon
    dilated_points.append(dilated_points[0])
    return np.array(dilated_points)
    

def get_cuboid_from_uuid(track_uuid, log_dir, timestamp = None):
    df = read_feather(log_dir / 'sm_annotations.feather')
    
    track_df = df[df["track_uuid"] == track_uuid]

    if timestamp:
        track_df = track_df[track_df["timestamp_ns"] == timestamp]
        if track_df.empty:
            print('Invalid timestamp does not exist for given track_uuid.')
            return None

    track_cuboids = CuboidList.from_dataframe(track_df)
    
    return track_cuboids[0]


def to_scenario_dict(object_datastructure, log_dir)->dict:

    if isinstance(object_datastructure, dict):
        object_dict = deepcopy(object_datastructure)
    elif isinstance(object_datastructure, list) or isinstance(object_datastructure, np.ndarray):
        object_dict = {uuid: unwrap_func(get_object)(uuid, log_dir) for uuid in object_datastructure}
    elif isinstance(object_datastructure, str):
        object_dict = {object_datastructure: unwrap_func(get_object)(object_datastructure, log_dir)}
    elif isinstance(object_datastructure, int):
        timestamp = object_datastructure
        df = read_feather(log_dir / 'sm_annotations.feather')
        timestamp_df = df[df['timestamp_ns'] == timestamp]

        if timestamp_df.empty:
            print(f'Timestamp {timestamp} not found in annotations')

        object_dict = {track_uuid: [timestamp] for track_uuid in timestamp_df['track_uuid'].unique()}
    else:
        print(f'Provided object, {object_datastructure}, of type {type(object_datastructure)}, must be a track_uuid, list[track_uuid], \
              timestamp, or dict[timestamp:list[timestamp]]')
        print('Comparing to all objects in the log.')

        df = read_feather(log_dir / 'sm_annotations.feather')
        all_uuids = df['track_uuid'].unique()
        object_dict, _ = parallelize_uuids(get_object, all_uuids, log_dir)
    
    return object_dict


@composable_relational
def being_crossed_by(
    track_candidates:Union[list,dict], 
    related_candidates:Union[list,dict], 
    log_dir:Path,
    direction:Literal["forward", "backward", "left", "right"]="forward",
    in_direction:Literal['clockwise','counterclockwise','either']='either',
    forward_thresh:float=10,
    lateral_thresh:float=5)->dict:
    """
    Identifies objects that are being crossed by one of the related candidate objects. A crossing is defined as
    the related candidate's centroid crossing the half-midplane of a tracked candidate. The direction of the half-
    midplane is specified with the direction. 

    Args:
        track_candidates: Tracks to analyze .
        related_candidates: Candidates (e.g., pedestrians or vehicles) to check for crossings.
        log_dir: Path to scenario logs.
        direction: specifies the axis and direction the half midplane extends from 
        in_direction: which direction the related candidate has to cross the midplane for it to be considered a crossing
        forward_thresh: how far the midplane extends from the edge of the tracked object
        lateral_thresh: the two planes offset from the midplane. If an related candidate crosses the midplane, it will 
        continue being considered crossing until it goes past the lateral_thresh.

    Returns:
        A filtered scenario dictionary containing all of the track candidates that were crossed by 
        the related candidates given the specified constraints.

    Example:
        overtaking_on_left = being_crossed_by(moving_cars, moving_cars, log_dir, direction="left", in_direction="clockwise", forward_thresh=4)
        vehicles_crossed_by_peds = being_crossed_by(vehicles, pedestrians, log_dir)
    """
    track_uuid = track_candidates
    VELOCITY_THRESH   = .2 #m/s 

    crossings = {}
    crossed_timestamps = []
    
    track = get_cuboid_from_uuid(track_uuid, log_dir)
    forward_thresh = track.length_m/2 + forward_thresh
    left_bound = -track.width_m/2
    right_bound = track.width_m/2

    for candidate_uuid in related_candidates:
        if candidate_uuid == track_uuid:
            continue

        #Transform from city to tracked_object coordinate frame
        candidate_pos, timestamps = get_nth_pos_deriv(candidate_uuid, 0, log_dir, coordinate_frame=track_uuid, direction=direction)
        candidate_vel, timestamps = get_nth_pos_deriv(candidate_uuid, 1, log_dir, coordinate_frame=track_uuid, direction=direction)

        for i in range(1,len(candidate_pos)):
            y0 = candidate_pos[i-1, 1]
            y1 = candidate_pos[i, 1]
            y_vel = candidate_vel[i, 1]
            if ((y0<left_bound<y1 or y1<right_bound<y0 or y0<right_bound<y1 or y1<left_bound<y0)
            and abs(y_vel) > VELOCITY_THRESH) and (track.length_m/2<=candidate_pos[i,0]<=forward_thresh) \
            and candidate_uuid != track_uuid:
                
                #1 if moving right, -1 if moving left
                direction = (y1-y0)/abs(y1-y0)
                start_index = i-1
                end_index = i
                updated = True
                
                if (direction == 1 and in_direction == 'clockwise'
                or direction == -1 and in_direction == 'counterclockwise'):
                    #The object is not moving in the specified crossing direction
                    continue
        
                while updated:
                    updated = False
                    if start_index>=0 and direction*candidate_pos[start_index, 1] < lateral_thresh \
                    and direction*candidate_vel[start_index,1] > VELOCITY_THRESH:
                        if candidate_uuid not in crossings:
                            crossings[candidate_uuid] = []
                        crossings[candidate_uuid].append(timestamps[start_index])
                        crossed_timestamps.append(timestamps[start_index])
                        updated = True
                        start_index -= 1

                    if end_index < len(timestamps) and direction*candidate_pos[end_index, 1] < lateral_thresh \
                    and direction*candidate_vel[end_index, 1] > VELOCITY_THRESH:
                        if candidate_uuid not in crossings:
                            crossings[candidate_uuid] = []
                        crossings[candidate_uuid].append(timestamps[end_index])
                        crossed_timestamps.append(timestamps[end_index])
                        updated = True
                        end_index += 1

    return crossed_timestamps, crossings


def plot_lane_segments(
    ax: matplotlib.axes.Axes, lane_segments: list[LaneSegment], lane_color: np.ndarray = np.array([.2,.2,.2])
) -> None:
    """
    Args:
        ax:
        lane_segments:
    """
    for ls in lane_segments:
        pts_city = ls.polygon_boundary
        ALPHA = 1.0  # 0.1
        vector_plotting_utils.plot_polygon_patch_mpl(
            polygon_pts=pts_city, ax=ax, color=lane_color, alpha=ALPHA, zorder=1
        )

        for bound_type, bound_city in zip(
            [ls.left_mark_type, ls.right_mark_type], [ls.left_lane_boundary, ls.right_lane_boundary]
        ):
            if "YELLOW" in bound_type:
                mark_color = "y"
            elif "WHITE" in bound_type:
                mark_color = "w"
            else:
                mark_color = "grey"  # "b" lane_color #

            LOOSELY_DASHED = (0, (5, 10))

            if "DASHED" in bound_type:
                linestyle = LOOSELY_DASHED
            else:
                linestyle = "solid"

            if "DOUBLE" in bound_type:
                left, right = polyline_utils.get_double_polylines(
                    polyline=bound_city.xyz[:, :2], width_scaling_factor=0.1
                )
                ax.plot(left[:, 0], left[:, 1], mark_color, alpha=ALPHA, linestyle=linestyle, zorder=2)
                ax.plot(right[:, 0], right[:, 1], mark_color, alpha=ALPHA, linestyle=linestyle, zorder=2)
            else:
                ax.plot(
                    bound_city.xyz[:, 0],
                    bound_city.xyz[:, 1],
                    mark_color,
                    alpha=ALPHA,
                    linestyle=linestyle,
                    zorder=2,
                )


def plot_polygon_patch_pv(polygon_pts, plotter: pv.plotter, color, opacity):
    num_points = len(polygon_pts) - 1
    faces = np.array(range(-1, num_points))
    faces[0] = num_points

    poly_data = pv.PolyData(polygon_pts)
    poly_data.faces = faces

    # Add the polygon to the plotter
    plotter.add_mesh(poly_data, color=color, opacity=opacity, lighting=False)


def plot_lane_segments_pv(
    plotter: pv.Plotter, lane_segments: list[LaneSegment], lane_color:np.ndarray = np.array([.2,.2,.2])
) -> list[vtk.vtkActor]:
    """
    Args:
        plotter: PyVista Plotter instance.
        lane_segments: List of LaneSegment objects to plot.
        lane_color: Color for the lane polygons.
    """
    actors = []
    for ls in lane_segments:
        pts_city = ls.polygon_boundary
        ALPHA = 0  # Adjust opacity

        # Add polygon boundary
        plot_polygon_patch_pv(
            polygon_pts=pts_city, plotter=plotter, color=lane_color, opacity=ALPHA)

        for bound_type, bound_city in zip(
            [ls.left_mark_type, ls.right_mark_type], [ls.left_lane_boundary, ls.right_lane_boundary]
        ):
            if "YELLOW" in bound_type:
                mark_color = "yellow"
            elif "WHITE" in bound_type:
                mark_color = "gray"
            else:
                mark_color = "black"

            if "DASHED" in bound_type:
                line_style = "dashed"  # PyVista doesn't support direct line styles; this is conceptual
            else:
                line_style = "solid"

            if "DOUBLE" in bound_type:
                left, right = polyline_utils.get_double_polylines(
                    polyline=bound_city.xyz, width_scaling_factor=0.1
                )
                plotter.add_lines(left, color=mark_color, width=2, connected=True)
                plotter.add_lines(right, color=mark_color, width=2, connected=True)
            else:
                plotter.add_lines(
                    bound_city.xyz, color=mark_color, width=2, connected=True)

    return actors


def plot_map(log_dir:Path, save_plot: bool = False) -> None:
    """
    Visualize both ego-vehicle poses and the per-log local vector map.

    Crosswalks are plotted in purple. Lane segments plotted in dark gray.
    """

    fig = plt.figure(1, figsize=(10, 10))
    ax = fig.add_subplot(111)
    avm = get_map(log_dir)

    # scaled to [0,1] for matplotlib.
    PURPLE_RGB = [201, 71, 245]
    PURPLE_RGB_MPL = np.array(PURPLE_RGB) / 255

    crosswalk_color = PURPLE_RGB_MPL
    CROSSWALK_ALPHA = 0.6
    for pc in avm.get_scenario_ped_crossings():
        vector_plotting_utils.plot_polygon_patch_mpl(
            polygon_pts=pc.polygon[:, :2],
            ax=ax,
            color=crosswalk_color,
            alpha=CROSSWALK_ALPHA,
            zorder=3,
        )

    plot_lane_segments(ax=ax, lane_segments=avm.get_scenario_lane_segments())

    plt.title(f"Log {log_dir.name}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    if save_plot:
        plt.savefig(f'output/{log_dir.name}_map.png')


@composable_relational
def near_objects(
    track_uuid:Union[list,dict], 
    candidate_uuids:Union[list,dict], 
    log_dir:Path,
    distance_thresh:float=10, 
    min_objects:float=1)->dict:
    """
    Identifies timestamps when a tracked object is near a specified set of related objects.

    Args:
        track_candidates: Tracks to analyze (list of UUIDs or scenario dictionary).
        related_candidates: Candidates to check for proximity (list of UUIDs or scenario dictionary).
        log_dir: Path to scenario logs.
        distance_thresh: Maximum distance in meters a related candidate can be away to be considered "near".
        min_objects: Minimum number of related objects required to be near the tracked object.
            If None, the tracked object must be near all related objects.

    Returns:
        dict: 
            A scenario dictionary where:
            Keys are timestamps when the tracked object is near the required number of related objects.
            Values are lists of related candidate UUIDs present at those timestamps.

    Example:
        vehicles_near_ped_group = near_objects(vehicles, pedestrians, log_dir, min_objects=3)
    """

    if not min_objects:
        min_objects = len(candidate_uuids)

    near_objects_dict = {}
    for candidate in candidate_uuids:
        if candidate == track_uuid:
            continue

        traj, timestamps = get_nth_pos_deriv(candidate, 0, log_dir, coordinate_frame=track_uuid)
        timestamps = timestamps[np.where(np.linalg.norm(traj, axis=1) < distance_thresh)]

        for timestamp in timestamps:
            if timestamp not in near_objects_dict:
                near_objects_dict[timestamp] = []
            near_objects_dict[timestamp].append(candidate)
        
    timestamps = []
    keys = list(near_objects_dict.keys())
    for timestamp in keys:
        if len(near_objects_dict[timestamp]) >= min_objects:
            timestamps.append(timestamp)
        else:
            near_objects_dict.pop(timestamp)

    near_objects_dict = swap_keys_and_listed_values(near_objects_dict)

    return timestamps, near_objects_dict


def plot_map_pv(avm:ArgoverseStaticMap, plotter:pv.Plotter) -> list[vtk.vtkActor]:
    actors = []

    for pc in avm.get_scenario_ped_crossings():
        ped_crossing_mesh = pv.PolyData(pc.polygon)
        faces = np.array([4, 0, 1, 2, 3])  # The number of vertices in the polygon followed by the indices of the points
        # Add faces to the PolyData
        ped_crossing_mesh.faces = faces
        pc_actor = plotter.add_mesh(ped_crossing_mesh, color='purple', opacity=.3, lighting=False, show_vertices=False)
        actors.append(pc_actor)

    lane_segments = avm.get_scenario_lane_segments()
    lane_actors = plot_lane_segments_pv(plotter, lane_segments)
    actors.extend(lane_actors)

    return actors


def visualize_scenario(scenario:dict, log_dir:Path, output_dir:Path, with_intro=True, description='scenario visualization',
                        with_map=True, with_cf=False, with_lidar=False, relationship_edges=True, stride=1, av2_log_dir=None):
    """
    Generate a birds-eye-view video of a series of LiDAR scans.
    
    :param lidar_files: List of file paths to LiDAR scan data.
    :param output_file: Path to the output video file.
    """

    #Conversion to legacy code
    scenario_dict = reconstruct_track_dict(scenario)
    relationship_dict = reconstruct_relationship_dict(scenario)

    pv.start_xvfb() #
    FPS = 10
    output_file = output_dir / (description + '_n' + str(len(scenario_dict)) + '.mp4')
    plotter = pv.Plotter(off_screen=True)
    plotter.open_movie(output_file, framerate=FPS)

    if av2_log_dir is None:
        split = get_log_split(log_dir)
        av2_log_dir = AV2_DATA_DIR / split / log_dir.name

    dataset = AV2SensorDataLoader(data_dir=av2_log_dir.parent, labels_dir=av2_log_dir.parent)
    log_id = log_dir.name

    set_camera_position_pv(plotter, scenario_dict, relationship_dict, log_dir)
    plotter.add_legend([(description,'black'),(log_dir.name,'black')],
                        bcolor='white', border=True, loc='upper left',size=(.7,.1))

    if with_map:
        avm = get_map(log_dir)
        plot_map_pv(avm, plotter)

    if with_lidar:
        lidar_paths = dataset.get_ordered_log_lidar_fpaths(log_id)

    if with_intro:
        plot_visualization_intro(plotter, scenario_dict, log_dir, relationship_dict, description=description)

    scenario_objects = swap_keys_and_listed_values(scenario_dict)
    related_dict = key_by_timestamps(relationship_dict)

    ego_uuid = get_ego_uuid(log_dir)
    df = read_feather(log_dir / 'sm_annotations.feather')
    ego_df = df[df['track_uuid'] == ego_uuid]
    timestamps = sorted(ego_df['timestamp_ns'])
    frequency = 1/((timestamps[1] - timestamps[0])/1E9)

    for i in range(0, len(timestamps), stride):
        #print(f'{i}/{len(timestamps)}', end='\r')
        # Load LiDAR scan
        timestamp = timestamps[i]
        ego_to_city = dataset.get_city_SE3_ego(log_id, timestamp)

        timestamp_df = df[df['timestamp_ns'] == timestamp]
        timestamp_actors = []

        if scenario_objects and timestamp in scenario_objects:
            scenario_cuboids = timestamp_df[timestamp_df['track_uuid'].isin(scenario_objects[timestamp])]
            scenario_cuboids = CuboidList.from_dataframe(scenario_cuboids)
 
            related_uuids = []
            relationship_edge_mesh = None      
            if related_dict and timestamp in related_dict:
                for track_uuid, related_uuid_list in related_dict[timestamp].items():

                    new_related_uuids = set(related_uuid_list).difference(scenario_objects[timestamp])
                    related_uuids.extend(new_related_uuids)

                    if relationship_edges:
                        relationship_edge_mesh = append_relationship_edges(relationship_edge_mesh,
                                                track_uuid, related_uuid_list, log_dir, timestamp, ego_to_city)

            if relationship_edges and relationship_edge_mesh:
                edges_actor = plotter.add_mesh(relationship_edge_mesh, color='black', line_width=2)
                timestamp_actors.append(edges_actor)

            related_df = timestamp_df[timestamp_df['track_uuid'].isin(related_uuids)]
            related_cuboids = CuboidList.from_dataframe(related_df)

            all_timestamp_uuids = timestamp_df['track_uuid'].unique()
            referred_or_related_uuids = set(scenario_objects[timestamp]).union(related_uuids)


            if ego_uuid not in referred_or_related_uuids:
                ego_cuboid = CuboidList.from_dataframe(timestamp_df[timestamp_df['track_uuid'] == ego_uuid])
                referred_or_related_uuids.add(ego_uuid)
            else:
                ego_cuboid = []

            other_uuids = set(all_timestamp_uuids).difference(referred_or_related_uuids)
            other_cuboids = timestamp_df[timestamp_df['track_uuid'].isin(other_uuids)]
            other_cuboids = CuboidList.from_dataframe(other_cuboids)
        else:
            scenario_cuboids = []
            related_cuboids = []
            ego_cuboid = CuboidList.from_dataframe(timestamp_df[timestamp_df['track_uuid'] == ego_uuid])
            other_cuboids = CuboidList.from_dataframe(timestamp_df[timestamp_df['track_uuid'] != ego_uuid])

        # Add new point cloud
        if with_lidar:
            sweep = Sweep.from_feather(lidar_paths[i])
            scan = sweep.xyz
            scan_city = ego_to_city.transform_from(scan)
            scan_actor = plotter.add_mesh(scan_city, color='gray', point_size=1)
            timestamp_actors.append(scan_actor)
        
        # Add new cuboids
        scenario_actors = plot_cuboids(scenario_cuboids, plotter, ego_to_city, with_label=True, color='lime', opacity=1, with_cf=with_cf)
        related_actors = plot_cuboids(related_cuboids, plotter, ego_to_city, with_label=True, color='blue', opacity=1, with_cf=with_cf)
        other_actors = plot_cuboids(other_cuboids, plotter, ego_to_city, color='red', opacity=1,  with_cf=with_cf)
        ego_actor = plot_cuboids(ego_cuboid, plotter, ego_to_city, color='red', with_label=True, opacity=1, with_cf=with_cf)

        timestamp_actors.extend(scenario_actors)
        timestamp_actors.extend(related_actors)
        timestamp_actors.extend(other_actors)
        timestamp_actors.extend(ego_actor)

        # Render and write frame
        if i == 0 and with_intro:
            animate_legend_intro(plotter)

        num_frames = max(1,int(FPS//frequency))
        for _ in range(num_frames):
            plotter.write_frame()

        # Add point cloud to the plotter
        plotter.remove_actor(timestamp_actors)

    # Finalize and close movie
    plotter.close()
    print(f'Scenario "{description}" visualized successfully!')


def set_camera_position_pv(plotter:pv.Plotter, scenario_dict:dict, relationship_dict:dict, log_dir):

    scenario_height = -np.inf

    #Ego vehicle should always be in the camera's view
    ego_pos, timestamps = get_nth_pos_deriv(get_ego_uuid(log_dir), 0, log_dir)
    bl_corner = np.array([min(ego_pos[:,0]),min(ego_pos[:,1])])
    tr_corner = np.array([max(ego_pos[:,0]),max(ego_pos[:,1])])
    scenario_height = max(ego_pos[:,2])

    for track_uuid, scenario_timestamps in scenario_dict.items():
        if len(scenario_timestamps) == 0:
            continue

        pos, timestamps = get_nth_pos_deriv(track_uuid, 0, log_dir)
        scenario_pos = pos[np.isin(timestamps, scenario_timestamps)]

        if scenario_pos.any():
            track_bl_corner = np.min(scenario_pos[:,:2], axis=0)
            track_tr_corner = np.max(scenario_pos[:,:2], axis=0)
            scenario_height = max(scenario_height, np.max(scenario_pos[:,2]))

            bl_corner = np.min(np.vstack((bl_corner, track_bl_corner)), axis=0)
            tr_corner = np.max(np.vstack((tr_corner, track_tr_corner)), axis=0)
            scenario_height = max(scenario_height, np.max(scenario_pos[:,2]))

    if not dict_empty(relationship_dict):
        for track_uuid, related_objects in relationship_dict.items():
            for related_uuid, timestamps in related_objects.items():
                if len(timestamps) == 0:
                    continue

                pos, timestamps = get_nth_pos_deriv(related_uuid, 0, log_dir)
                scenario_pos = pos[np.isin(timestamps, scenario_timestamps)]

                if scenario_pos.any():
                    track_bl_corner = np.min(scenario_pos[:,:2], axis=0)
                    track_tr_corner = np.max(scenario_pos[:,:2], axis=0)

                    bl_corner = np.min(np.vstack((bl_corner, track_bl_corner)), axis=0)
                    tr_corner = np.max(np.vstack((tr_corner, track_tr_corner)), axis=0)
                    scenario_height = max(scenario_height, np.max(scenario_pos[:,2]))

    scenario_center = np.concatenate(((tr_corner+bl_corner)/2, [scenario_height]))
    height_above_scenario = 1.1*(np.linalg.norm(tr_corner-bl_corner))/(2*np.tan(np.deg2rad(plotter.camera.view_angle)/2))

    camera_height = min(max(scenario_height+height_above_scenario, scenario_height+100), scenario_height+400)
    plotter.camera_position = [tuple(scenario_center+[0,0,camera_height]), (scenario_center), (0, 1, 0)]
     

def append_relationship_edges(relationship_edge_mesh:pv.PolyData, track_uuid, related_uuids, log_dir, timestamp, transform:SE3):
    df = read_feather(log_dir / 'sm_annotations.feather')
    track_df = df[df['track_uuid'] == track_uuid]
    timestamped_track = track_df[track_df['timestamp_ns'] == timestamp]
    track_pos = timestamped_track[['tx_m', 'ty_m', 'tz_m']].to_numpy()
    track_pos = transform.transform_from(track_pos)

    for related_uuid in related_uuids:
        related_df = df[df['track_uuid'] == related_uuid]
        timestamped_related = related_df[related_df['timestamp_ns'] == timestamp]
        related_pos = timestamped_related[['tx_m', 'ty_m', 'tz_m']].to_numpy()
        related_pos = transform.transform_from(related_pos)

        points = np.vstack((track_pos, related_pos))
        line = np.array([2,0,1])
        if relationship_edge_mesh == None:
            relationship_edge_mesh = pv.PolyData(points, lines=line)
        else:
            relationship_edge_mesh = relationship_edge_mesh.append_polydata(pv.PolyData(points, lines=line))

    return relationship_edge_mesh


def animate_legend_intro(plotter:pv.plotter):
    plotter.add_legend([(' referred objects','lime', 'rectangle'),(' related objects','blue', 'rectangle'), (' other objects', 'red', 'rectangle'), ('3', 'red')],
            bcolor='white', border=True, loc='upper left',size=(.267,.13))
    
    for j in range(3):
        plotter.write_frame()

    plotter.add_legend([(' referred objects','lime', 'rectangle'),(' related objects','blue', 'rectangle'), (' other objects', 'red', 'rectangle'), (' 2', 'orange')],
            bcolor='white', border=True, loc='upper left',size=(.267,.13))
    
    for j in range(3):
        plotter.write_frame()        

    plotter.add_legend([(' referred objects','lime', 'rectangle'),(' related objects','blue', 'rectangle'), (' other objects', 'red', 'rectangle'), ('  1', 'yellow')],
            bcolor='white', border=True, loc='upper left',size=(.267,.13))
    
    for j in range(3):
        plotter.write_frame()
    plotter.add_legend([(' referred objects','lime', 'rectangle'),(' related objects','blue', 'rectangle'), (' other objects', 'red', 'rectangle'), ('   GO!', 'green')],
            bcolor='white', border=True, loc='upper left',size=(.267,.13))
    
    for j in range(3):
        plotter.write_frame()

    plotter.add_legend([(' referred objects','lime', 'rectangle'),(' related objects','blue', 'rectangle'), (' other objects', 'red', 'rectangle')],
        bcolor='white', border=True, loc='upper left',size=(.2,.1))


def plot_visualization_intro(plotter: pv.Plotter, scenario_dict:dict, log_dir, related_dict:dict[str,dict]={}, description='scenario visualization'):
    
    track_first_appearences = {}
    for track_uuid, timestamps in scenario_dict.items():
        if timestamps:
            track_first_appearences[track_uuid] = min(timestamps)

    related_first_appearances = {}
    for track_uuid, related_objects in related_dict.items():
        for related_uuid, timestamps in related_objects.items():
            if timestamps and related_uuid in related_first_appearances:
                related_first_appearances[related_uuid] = min(min(timestamps),related_first_appearances[related_uuid])
            elif timestamps and (
            related_uuid not in track_first_appearences or min(timestamps) != track_first_appearences[related_uuid]):
                related_first_appearances[related_uuid] = min(timestamps)

    scenario_cuboids = []
    scenario_transforms = []
    related_cuboids = []
    related_transforms = []

    ego_poses = get_ego_SE3(log_dir)
    df = read_feather(log_dir / 'sm_annotations.feather')

    for track_uuid, timestamp in track_first_appearences.items():
        track_df = df[df['track_uuid'] == track_uuid]
        cuboid_df = track_df[track_df['timestamp_ns'] == timestamp]
        scenario_cuboids.extend(CuboidList.from_dataframe(cuboid_df))
        scenario_transforms.append(ego_poses[timestamp])

    for related_uuid, timestamp in related_first_appearances.items():
        track_df = df[df['track_uuid'] == related_uuid]
        cuboid_df = track_df[track_df['timestamp_ns'] == timestamp]
        related_cuboids.extend(CuboidList.from_dataframe(cuboid_df))
        related_transforms.append(ego_poses[timestamp])

    plotter.add_legend([(description,'black'),(log_dir.name,'black')],
                        bcolor='white', border=True, loc='upper left',size=(.7,.1))

    for i in range(4):
        actors = []

        actors.extend(plot_cuboids(scenario_cuboids, plotter, scenario_transforms, color='lime', opacity=1))
        actors.extend(plot_cuboids(related_cuboids, plotter, related_transforms, color='blue', opacity=1))

        for j in range(5):
            plotter.write_frame()

        plotter.remove_actor(actors)

        for j in range(5):
            plotter.write_frame()


def swap_keys_and_listed_values(dict:dict[float,list])->dict[float,list]:
    
    swapped_dict = {}
    for key, timestamp_list in dict.items():
        for timestamp in timestamp_list:
            if timestamp not in swapped_dict:
                swapped_dict[timestamp] = []
            swapped_dict[timestamp].append(key)

    return swapped_dict


def key_by_timestamps(dict:dict[str,dict[str,list[float]]]) -> dict[float,dict[str,list[str]]]:
    if not dict:
        return {}

    temp = deepcopy(dict)

    for track_uuid in temp.keys():
        temp[track_uuid] = swap_keys_and_listed_values(temp[track_uuid])

    swapped_dict = {}
    for track_uuid, timestamp_dict in temp.items():
        for timestamp in timestamp_dict.keys():

            if timestamp not in swapped_dict:
                swapped_dict[timestamp] = {}
            if track_uuid not in swapped_dict[timestamp]:
                swapped_dict[timestamp][track_uuid] = []

            swapped_dict[timestamp][track_uuid] = temp[track_uuid][timestamp]

    return swapped_dict


def dict_empty(d:dict):
    if len(d) == 0:
        return True

    for  value in d.values():
        if isinstance(value, list) and len(value) > 0:
            return False

        if isinstance(value, dict) and not dict_empty(value):
            return False
        
    return True


@composable_relational
def following(
    track_uuid:Union[list,dict],
    candidate_uuids:Union[list,dict],
    log_dir:Path) -> tuple[list, dict[str,list]]:
    """
    Returns timestamps when the tracked object is following a lead object
    Following is defined simultaneously moving in the same direction and lane,
    """

    lead_timestamps = []
    leads = {}

    avm = get_map(log_dir)
    track_lanes = get_scenario_lanes(track_uuid, log_dir, avm=avm)
    track_vel, track_timestamps = get_nth_pos_deriv(track_uuid, 1, log_dir, coordinate_frame=track_uuid)

    track_cuboid = get_cuboid_from_uuid(track_uuid, log_dir)
    track_width = track_cuboid.width_m/2
    track_length = track_cuboid.length_m/2

    FOLLOWING_THRESH = 25 + track_length #m
    LATERAL_TRHESH = 5 #m
    HEADING_SIMILARITY_THRESH = .5 #cosine similarity

    for j, candidate in enumerate(candidate_uuids):
        if candidate == track_uuid:
            continue

        #print(f'{j}/{len(candidate_uuids)}', end='\r')
        candidate_pos, _ = get_nth_pos_deriv(candidate, 0, log_dir, coordinate_frame=track_uuid)
        candidate_vel, _ = get_nth_pos_deriv(candidate, 1, log_dir, coordinate_frame=track_uuid) 
        candidate_yaw, timestamps = get_nth_yaw_deriv(candidate, 0, log_dir, coordinate_frame=track_uuid)
        candidate_lanes = get_scenario_lanes(candidate, log_dir, avm=avm)

        overlap_track_vel = track_vel[np.isin(track_timestamps, timestamps)]
        candidate_heading_similarity = np.zeros(len(timestamps))

        candidate_cuboid = get_cuboid_from_uuid(track_uuid, log_dir)
        candidate_width = candidate_cuboid.width_m/2

        for i in range(len(timestamps)):

            if np.linalg.norm(candidate_vel[i]) > .5:
                candidate_heading = candidate_vel[i, :2]/np.linalg.norm(candidate_vel[i,:2])
            else:
                candidate_heading = np.array([np.cos(candidate_yaw[i]), np.sin(candidate_yaw[i])])

            if np.linalg.norm(overlap_track_vel[i]) > .5:
                track_heading = overlap_track_vel[i, :2]/np.linalg.norm(overlap_track_vel[i,:2])  
            else:
                #Coordinates are in track_coordinate frame.
                track_heading = np.array([1,0])

            candidate_heading_similarity[i] = np.dot(track_heading, candidate_heading)

        for i in range(len(timestamps)):
            if track_lanes[timestamps[i]] and candidate_lanes[timestamps[i]] \
            and (((track_lanes[timestamps[i]].id == candidate_lanes[timestamps[i]].id \
                or candidate_lanes[timestamps[i]].id in track_lanes[timestamps[i]].successors) \
                and track_length<candidate_pos[i, 0]<FOLLOWING_THRESH and -LATERAL_TRHESH<candidate_pos[i,1]<LATERAL_TRHESH \
                and candidate_heading_similarity[i] > HEADING_SIMILARITY_THRESH)\
            or (track_lanes[timestamps[i]].left_neighbor_id == candidate_lanes[timestamps[i]].id 
                or track_lanes[timestamps[i]].right_neighbor_id == candidate_lanes[timestamps[i]].id) \
                and track_length<candidate_pos[i, 0]<FOLLOWING_THRESH 
                and (-track_width<=candidate_pos[i,1]+candidate_width<=track_width or -track_width<=candidate_pos[i,1]-candidate_width<=track_width)\
                and candidate_heading_similarity[i] > HEADING_SIMILARITY_THRESH):

                if candidate not in leads:
                    leads[candidate] = []
                leads[candidate].append(timestamps[i]) 
                lead_timestamps.append(timestamps[i]) 
        
    return lead_timestamps, leads

@composable_relational
def heading_in_relative_direction_to(track_candidates, related_candidates, log_dir, direction:Literal['same', 'opposite', 'perpendicular']):
    """Returns the subset of track candidates that are traveling in the given direction compared to the related canddiates.

    Arguements:
        track_candidates: The set of objects that could be traveling in the given direction
        related_candidates: The set of objects that the direction is relative to
        log_dir: The path to the log data
        direction: The direction that the positive tracks are traveling in relative to the related candidates
            "opposite" indicates the track candidates are traveling in a direction 135-180 degrees from the direction the related candidates
            are heading toward.
            "same" indicates the track candidates that are traveling in a direction 0-45 degrees from the direction the related candiates
            are heading toward.
            "same" indicates the track candidates that are traveling in a direction 45-135 degrees from the direction the related candiates
            are heading toward.

    Returns:
        the subset of track candidates that are traveling in the given direction compared to the related candidates.

    Example:
        oncoming_traffic = traveling_relative_direction(vehicles, ego_vehicle, log_dir, direction='opposite')    
    """
    track_uuid = track_candidates

    track_pos, _ = get_nth_pos_deriv(track_uuid, 0, log_dir)
    track_vel, track_timestamps = get_nth_pos_deriv(track_uuid, 1, log_dir)

    traveling_in_direction_timestamps = []
    traveling_in_direction_objects = {}
    ego_to_city = get_ego_SE3(log_dir)

    for related_uuid in related_candidates:
        if track_uuid == related_uuid:
            continue
        
        related_pos, _ = get_nth_pos_deriv(related_uuid, 0, log_dir)
        related_vel, related_timestamps = get_nth_pos_deriv(related_uuid, 1, log_dir)
        for i, timestamp in enumerate(track_timestamps):

            if timestamp in related_timestamps:

                track_dir = track_vel[i]
                related_dir = related_vel[list(related_timestamps).index(timestamp)]

                if np.linalg.norm(track_dir) < 1 and has_free_will(track_uuid,log_dir) and np.linalg.norm(related_dir) > 1:
                    track_cuboid = get_cuboid_from_uuid(track_uuid, log_dir, timestamp=timestamp)
                    track_self_dir = np.array([1,0,0])

                    timestamp_track_pos = track_pos[i]
                    timestamp_track_posx = ego_to_city[timestamp].compose(track_cuboid.dst_SE3_object).transform_from(track_self_dir)
                    track_dir = timestamp_track_posx - timestamp_track_pos

                elif np.linalg.norm(related_dir) < 1 and has_free_will(related_uuid,log_dir) and np.linalg.norm(track_dir) > .5:
                    related_cuboid = get_cuboid_from_uuid(related_uuid, log_dir, timestamp=timestamp)
                    related_x_dir = np.array([1,0,0])
                    timestamp_related_pos = related_pos[list(related_timestamps).index(timestamp)]
                    timestamp_related_posx = ego_to_city[timestamp].compose(related_cuboid.dst_SE3_object).transform_from(related_x_dir)
                    related_dir = timestamp_related_posx - timestamp_related_pos
                elif np.linalg.norm(track_dir) < 1 or np.linalg.norm(related_dir) < 1:
                    continue
                
                track_dir = track_dir/np.linalg.norm(track_dir)
                related_dir = related_dir/np.linalg.norm(related_dir)
                angle = np.rad2deg(np.arccos(np.dot(track_dir, related_dir)))

                if (angle <= 45 and direction == 'same'
                or 45 < angle < 135 and direction == 'perpendicular'
                or 135 <= angle < 180 and direction == 'opposite'):
                    if related_uuid not in traveling_in_direction_objects:
                        traveling_in_direction_objects[related_uuid] = []
                    traveling_in_direction_objects[related_uuid].append(timestamp)
                    traveling_in_direction_timestamps.append(timestamp)

    return traveling_in_direction_timestamps, traveling_in_direction_objects
                


@composable
def stationary(track_candidates:Union[list,dict], log_dir:Path):
    """
    Returns objects that moved less than 2m over their length of observation in the scneario.
    This object is only intended to separate parked from active vehicles. 
    Use has_velocity() with thresholding if you want to indicate vehicles that are temporarily stopped.

    Args:
        track_candidates: Tracks to analyze (list of UUIDs or scenario dictionary).
        log_dir: Path to scenario logs.

    Returns:
        dict: 
            A filtered scenario dictionary where keys are track UUIDs and values are lists of timestamps when the object is stationary.

    Example:
        parked_vehicles = stationary(vehicles, log_dir)
    """
    track_uuid = track_candidates

    #Displacement threshold needed because of annotation jitter
    DISPLACMENT_THRESH = 2

    pos, timestamps = get_nth_pos_deriv(track_uuid, 0, log_dir)

    max_displacement = np.max(pos, axis=0) - np.min(pos, axis=0)

    if np.linalg.norm(max_displacement) < DISPLACMENT_THRESH:
        return list(timestamps)
    else:
        return []


@composable_relational
def __at_stop_signs(track_uuid, stop_sign_uuids, log_dir, forward_thresh=10) -> tuple[list, dict[str,list]]:
    RIGHT_THRESH = 7 #m

    stop_sign_timestamps = []
    stop_signs = {}
    
    track_lanes = get_scenario_lanes(track_uuid, log_dir)

    for stop_sign_id in stop_sign_uuids:
        pos, _ = get_nth_pos_deriv(track_uuid, 0, log_dir, coordinate_frame=stop_sign_id)
        yaws, timestamps = get_nth_yaw_deriv(track_uuid, 0, log_dir, coordinate_frame=stop_sign_id, in_degrees=True)
        for i in range(len(timestamps)):
            if (-1<pos[i,0]<forward_thresh and -RIGHT_THRESH<pos[i,1]<0 
            and track_lanes.get(timestamps[i],None) 
            and stop_sign_lane(stop_sign_id, log_dir) 
            and track_lanes[timestamps[i]].id == stop_sign_lane(stop_sign_id, log_dir).id
            and (yaws[i] >= 90 or yaws[i] <= -90)):

                if stop_sign_id not in stop_signs:
                    stop_signs[stop_sign_id] = []
                stop_signs[stop_sign_id].append(timestamps[i])
            
                if timestamps[i] not in stop_sign_timestamps:
                    stop_sign_timestamps.append(timestamps[i])

    return stop_sign_timestamps, stop_signs


def at_stop_sign(track_candidates:Union[list,dict], log_dir:Path, forward_thresh:float=10):
    """
    Identifies timestamps when a tracked object is in a lane corresponding to a stop sign. The tracked
    object must be within 15m of the stop sign. This may highlight vehicles using street parking near a stopped sign.

    Args:
        track_candidates: Tracks to analyze (list of UUIDs or scenario dictionary).
        log_dir: Path to scenario logs.
        forward_thresh: Distance in meters the vehcile is from the stop sign in the stop sign's front direction

    Returns:
        dict: 
            A filtered scenario dictionary where keys are track UUIDs and values are lists of timestamps when the object is at a stop sign.

    Example:
        vehicles_at_stop_sign = at_stop_sign(vehicles, log_dir)
    """

    stop_sign_uuids = get_uuids_of_category(log_dir, 'STOP_SIGN')
    return __at_stop_signs(track_candidates, stop_sign_uuids, log_dir, forward_thresh=forward_thresh)


@composable
def occluded(track_uuid, log_dir):

    annotations_df = read_feather(log_dir / 'sm_annotations.feather')
    track_df = annotations_df[annotations_df['track_uuid'] == track_uuid]
    track_when_occluded = track_df[track_df['num_interior_pts'] == 0]

    if track_when_occluded.empty:
        return []
    else:
        return sorted(track_when_occluded['timestamp_ns'])

def stop_sign_lane(stop_sign_id, log_dir) -> LaneSegment:
    avm = get_map(log_dir)
    pos, _ = get_nth_pos_deriv(stop_sign_id, 0, log_dir)

    ls_list = avm.get_nearby_lane_segments(pos[0,:2], 5)
    best_ls = None
    best_dist = np.inf
    for ls in ls_list:
        dist = np.linalg.norm(pos[0]-ls.right_lane_boundary.xyz[-1])

        if not ls.is_intersection and dist < best_dist:
            best_ls = ls

    if best_ls == None:
        for ls in ls_list:
            dist = np.linalg.norm(pos[0]-ls.right_lane_boundary.xyz[-1])

            if dist < best_dist:
                best_ls = ls

    if best_ls == None:
        print('Correct lane segment not found for stop sign!')
    
    return best_ls


def get_pos_within_lane(pos, ls: LaneSegment) -> tuple:

    if not ls or not is_point_in_polygon(pos[:2], ls.polygon_boundary[:,:2]):
        return None, None

    #Projecting to 2D for BEV
    pos = pos[:2]
    left_line = ls.left_lane_boundary.xyz[:,:2]
    right_line = ls.right_lane_boundary.xyz[:,:2]

    left_dist = 0
    left_point = None
    left_total_length = 0
    min_dist = np.inf
    for i in range(1, len(left_line)):
        segment_start = left_line[i-1]
        segment_end = left_line[i]

        segment_length = np.linalg.norm(segment_end-segment_start)
        segment_direction = (segment_end-segment_start)/segment_length
        segment_proj = np.dot((pos-segment_start), segment_direction)*segment_direction
        proj_length = np.linalg.norm(segment_proj)

        if 0 <= proj_length <= segment_length:
            proj_point = segment_start + segment_proj
        elif proj_length < 0:
            proj_point = segment_start
        else:
            proj_point = segment_end

        proj_dist = np.linalg.norm(pos-proj_point)

        if proj_dist < min_dist:
            min_dist = proj_dist
            left_point = segment_start + segment_proj
            left_dist = left_total_length + proj_length

        left_total_length += segment_length

    right_dist = 0
    right_point = None
    right_total_length = 0
    min_dist = np.inf
    for i in range(1, len(right_line)):
        segment_start = right_line[i-1]
        segment_end = right_line[i]

        segment_length = np.linalg.norm(segment_end-segment_start)
        segment_direction = (segment_end-segment_start)/segment_length
        segment_proj = np.dot((pos-segment_start), segment_direction)*segment_direction
        proj_length = np.linalg.norm(segment_proj)

        if 0 <= proj_length <= segment_length:
            proj_point = segment_start + segment_proj
        elif proj_length < 0:
            proj_point = segment_start
        else:
            proj_point = segment_end

        proj_dist = np.linalg.norm(pos-proj_point)

        if proj_dist < min_dist:
            min_dist = proj_dist
            right_point = segment_start + segment_proj
            right_dist = right_total_length + proj_length

        right_total_length += segment_length

    if left_point is not None and right_point is not None:
        total_length = (left_total_length + right_total_length)/2
        distance = (left_dist + right_dist)/2
        pos_along_length = distance/total_length

        total_width = np.linalg.norm(left_point - right_point)
        lateral_dir_vec = (left_point - right_point)/total_width
        lateral_proj = np.dot((pos-left_point), lateral_dir_vec)*lateral_dir_vec
        pos_along_width = np.linalg.norm(lateral_proj)/total_width
        return pos_along_length, pos_along_width
    
    else:
        print("Position not found within lane_segment. Debug function further.")
        return None, None


@composable
def in_drivable_area(track_candidates:Union[list,dict], log_dir:Path)->dict:
    """
    Identifies objects within track_candidates that are within a drivable area.

    Args:
        track_candidates: Tracks to analyze (list of UUIDs or scenario dictionary).
        log_dir: Path to scenario logs.

    Returns:
        dict: 
            A filtered scenario dictionary where keys are track UUIDs and values are lists of timestamps when the object is in a drivable area.

    Example:
        buses_in_drivable_area = in_drivable_area(buses, log_dir)
    """
    track_uuid = track_candidates

    avm = get_map(log_dir)
    pos, timestamps = get_nth_pos_deriv(track_uuid, 0, log_dir)

    drivable_timestamps = []
    drivable_areas = avm.get_scenario_vector_drivable_areas()

    for i in range(len(timestamps)):
        for da in drivable_areas:
            if is_point_in_polygon(pos[i, :2], da.xyz[:,:2]):
                drivable_timestamps.append(timestamps[i])
                break

    return drivable_timestamps


@composable 
def on_road(
    track_candidates:Union[list,dict], 
    log_dir:Path)->dict:
    """
    Identifies objects that are on a road or bike lane. 
    This function should be used in place of in_driveable_area() when referencing objects that are on a road. 
    The road does not include parking lots or other driveable areas connecting the road to parking lots.

    Args:
        track_candidates: Tracks to filter (list of UUIDs or scenario dictionary).
        log_dir: Path to scenario logs.

    Returns:
    The subset of the track candidates that are currently on a road.

    Example:
        animals_on_road = on_road(animals, log_dir)   
    """

    timestamps = []
    lanes_keyed_by_timetamp = get_scenario_lanes(track_candidates, log_dir)
    
    for timestamp, lanes in lanes_keyed_by_timetamp.items():
        if lanes is not None:
            timestamps.append(timestamp)

    return timestamps


@composable_relational
def in_same_lane(
    track_candidates:Union[list,dict],
    related_candidates:Union[list,dict], 
    log_dir:Path) -> dict:
    """"
    Identifies tracks that are in the same road lane as a related candidate. 

    Args:
        track_candidates: Tracks to filter (list of UUIDs or scenario dictionary)
        related_candidates: Potential objects that could be in the same lane as the track (list of UUIDs or scenario dictionary)
        log_dir: Path to scenario logs.

    Returns:
        dict: 
            A filtered scenario dictionary where keys are track UUIDs and values are lists of timestamps when the object is on a road lane.

    Example:
        bicycle_in_same_lane_as_vehicle = in_same_lane(bicycle, regular_vehicle, log_dir)    
    """

    track_uuid = track_candidates
    traj, timestamps = get_nth_pos_deriv(track_uuid, 0, log_dir)

    avm = get_map(log_dir)
    track_ls = get_scenario_lanes(track_uuid, log_dir, avm=avm, traj=traj, timestamps=timestamps)
    semantic_lanes = {timestamp:get_semantic_lane(track_ls[timestamp], log_dir, avm=avm) for timestamp in timestamps}

    same_lane_timestamps = []
    sharing_lanes = {}

    for i, related_uuid in enumerate(related_candidates):
        #print(f'{i}/{len(related_candidates)}')

        if related_uuid == track_uuid:
            continue

        related_ls = get_scenario_lanes(related_uuid, log_dir, avm=avm)

        for timestamp in timestamps:
            if (timestamp in related_ls and related_ls[timestamp] is not None and 
            related_ls[timestamp] in semantic_lanes[timestamp]):
                if related_uuid not in sharing_lanes:
                    sharing_lanes[related_uuid] = []
                
                same_lane_timestamps.append(timestamp)
                sharing_lanes[related_uuid].append(timestamp)

    return same_lane_timestamps, sharing_lanes

@composable
def in_region_of_interest(track_uuid, log_dir):

    in_roi_timestamps = []

    avm = get_map(log_dir)
    timestamps = get_timestamps(track_uuid, log_dir)
    ego_poses = get_ego_SE3(log_dir)

    for timestamp in timestamps:
        cuboid = get_cuboid_from_uuid(track_uuid, log_dir, timestamp=timestamp)
        ego_to_city = ego_poses[timestamp]
        city_cuboid = cuboid.transform(ego_to_city)
        city_vertices = city_cuboid.vertices_m
        city_vertices = city_vertices.reshape(-1, 3)[:,:2]
        is_within_roi = avm.get_raster_layer_points_boolean(city_vertices, layer_name="ROI")
        if is_within_roi.any():
            in_roi_timestamps.append(timestamp)

    return in_roi_timestamps


def stopping(track_candidates, log_dir):

    track_uuid = track_candidates
    stopping_timestamps = []

    INITIAL_VEL_THRESH = 1.5 #m/s
    STOPPING_THRESH = .5 #m/s
    track_vel, _ = get_nth_pos_deriv(track_uuid, 1, log_dir)
    track_accel, timestamps = get_nth_pos_deriv(track_uuid, 2, log_dir)

    left = 0
    right = 0
    for i, timestamp in enumerate(timestamps):

        if track_vel[i] < STOPPING_THRESH:

            seg_timestamps = []
            while track_accel[right] < 0 and right >= left:
                seg_timestamps.append(timestamp)
                right -= 1
            right += 1

            if track_vel[right] > INITIAL_VEL_THRESH:
                stopping_timestamps.extend(seg_timestamps)

            left = i+1
            right = i+1
        right += 1

    return stopping_timestamps


def scenario_not(func):
    """
    Wraps composable functions to return the difference of the input track dict and output scenario dict. 
    Using scenario_not with a composable relational function will not return any relationships. 

    Args:
        composable_func: Any function that takes track_candidates as its first input

    Returns:

    Example:
        active_vehicles = scenario_not(stationary)(vehicles, log_dir)
    """
    def wrapper(track_candidates, *args, **kwargs):

        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        
        # Determine the position of 'log_dir'
        if 'log_dir' in params:
            log_dir_index = params.index('log_dir') - 1
        else:
            raise ValueError("The function does not have 'log_dir' as a parameter.")

        log_dir = args[log_dir_index]

        if func.__name__ == 'get_objects_in_relative_direction':
            track_dict = to_scenario_dict(args[0], log_dir)
        else:
            track_dict = to_scenario_dict(track_candidates, log_dir)

        if log_dir_index == 0:
            scenario_dict = func(track_candidates, log_dir, *args[1:], **kwargs)
        elif log_dir_index == 1:
            #composable_relational function
            scenario_dict = func(track_candidates, args[0], log_dir, *args[2:], **kwargs)

        remove_empty_branches(scenario_dict)
        not_dict = {track_uuid: [] for track_uuid in track_dict.keys()}

        for uuid in not_dict:
            if uuid in scenario_dict:
                not_timestamps = list(set(
                    get_scenario_timestamps(track_dict[uuid])).difference(get_scenario_timestamps(scenario_dict[uuid])))
                
                not_dict[uuid] = scenario_at_timestamps(track_dict[uuid], not_timestamps)
            else:
                not_dict[uuid] = track_dict[uuid]

        return not_dict
    return wrapper


@cache_manager.create_cache('scenario_and')
def scenario_and(scenario_dicts:list[dict])->dict:
    """
    Returns a composed scenario where the track objects are the intersection of all of the track objects
    with the same uuid and timestamps.

    Args:
        scenario_dicts: the scenarios to combine 

    Returns:
        dict:
            a filtered scenario dictionary that contains tracked objects found in all given scenario dictionaries
    
    Example:
        jaywalking_peds = scenario_and([peds_on_road, peds_not_on_pedestrian_crossing])

    """
    composed_dict = {}

    composed_track_dict = deepcopy(reconstruct_track_dict(scenario_dicts[0]))
    for i in range(1, len(scenario_dicts)):
        scenario_dict = scenario_dicts[i]
        track_dict = reconstruct_track_dict(scenario_dict)
        
        for track_uuid, timestamps in track_dict.items():
            if track_uuid not in composed_track_dict:
                continue

            composed_track_dict[track_uuid] = sorted(set(composed_track_dict[track_uuid]).intersection(timestamps))

        for track_uuid in list(composed_track_dict.keys()):
            if track_uuid not in track_dict:
                composed_track_dict.pop(track_uuid)

    for track_uuid, intersecting_timestamps in composed_track_dict.items():
        for scenario_dict in scenario_dicts:
            if track_uuid not in composed_dict:
                composed_dict[track_uuid] = scenario_at_timestamps(scenario_dict[track_uuid], intersecting_timestamps)
            else:
                related_children =  scenario_at_timestamps(scenario_dict[track_uuid],intersecting_timestamps)

                if isinstance(related_children, dict) and isinstance(composed_dict[track_uuid], dict):
                    composed_dict[track_uuid] = scenario_or([composed_dict[track_uuid], related_children])
                elif isinstance(related_children, dict) and not isinstance(composed_dict[track_uuid], dict):
                    related_children[track_uuid] = composed_dict[track_uuid]
                    composed_dict[track_uuid] = related_children
                elif not isinstance(related_children, dict) and isinstance(composed_dict[track_uuid], dict):
                    composed_dict[track_uuid][track_uuid] = related_children
                else:
                    composed_dict[track_uuid] = set(composed_dict[track_uuid]).intersection(related_children)

    return composed_dict
                    

@cache_manager.create_cache('scenario_or')
def scenario_or(scenario_dicts:list[dict]):
    """
    Returns a composed scenario where that tracks all objects and relationships in all of the input scenario dicts.

    Args:
        scenario_dicts: the scenarios to combine 

    Returns:
        dict:
            an expanded scenario dictionary that contains every tracked object in the given scenario dictionaries
    
    Example:
        be_cautious_around = scenario_or([animal_on_road, stroller_on_road])
    """

    composed_dict = deepcopy(scenario_dicts[0])
    for i in range(1, len(scenario_dicts)):
        for track_uuid, child in scenario_dicts[i].items():
            if track_uuid not in composed_dict:
                composed_dict[track_uuid] = child
            elif isinstance(child, dict) and isinstance(composed_dict[track_uuid], dict):
                composed_dict[track_uuid] = scenario_or([composed_dict[track_uuid], child])
            elif isinstance(child, dict) and not isinstance(composed_dict[track_uuid], dict):
                child[track_uuid] = composed_dict[track_uuid]
                composed_dict[track_uuid] = child
            elif not isinstance(child, dict) and isinstance(composed_dict[track_uuid], dict):
                composed_dict[track_uuid][track_uuid] = child
            else:
                composed_dict[track_uuid] = set(composed_dict[track_uuid]).union(child)

    return composed_dict


def remove_empty_branches(scenario_dict):
    
    if isinstance(scenario_dict, dict):
        track_uuids = list(scenario_dict.keys())
        for track_uuid in track_uuids:
            children = scenario_dict[track_uuid]
            timestamps = get_scenario_timestamps(children)
            if len(timestamps) == 0:
                scenario_dict.pop(track_uuid)
            else:
                remove_empty_branches(children)


def reverse_relationship(func):
    """
    Wraps relational functions to switch the top level tracked objects and relationships formed by the function. 

    Args:
        relational_func: Any function that takes track_candidates and related_candidates as its first and second arguements

    Returns:
        dict:
            scenario dict with swapped top-level tracks and related candidates

    Example:
        group_of_peds_near_vehicle = reverse_relationship(near_objects)(vehicles, peds, log_dir, min_objects=3)
    """
    def wrapper(track_candidates, related_candidates, log_dir, *args, **kwargs):

        if func.__name__ == 'get_objects_in_relative_direction':
            return has_objects_in_relative_direction(track_candidates, related_candidates, log_dir, *args, **kwargs)

        track_dict = to_scenario_dict(track_candidates, log_dir)
        related_dict = to_scenario_dict(related_candidates, log_dir)
        remove_empty_branches(track_dict)
        remove_empty_branches(related_dict)

        scenario_dict:dict = func(track_dict, related_dict, log_dir, *args, **kwargs)
        remove_empty_branches(scenario_dict)

        #Look for new relationships
        tc_uuids = list(track_dict.keys())
        rc_uuids = list(related_dict.keys())

        new_relationships = []
        for track_uuid, related_objects in scenario_dict.items():
            for related_uuid in related_objects.keys():
                if track_uuid in tc_uuids and related_uuid in rc_uuids \
                or track_uuid in rc_uuids and related_uuid in tc_uuids \
                and track_uuid != related_uuid:
                    new_relationships.append((track_uuid, related_uuid))

        #Reverese the scenario dict using these new relationships
        reversed_scenario_dict = {}
        for track_uuid, related_uuid in new_relationships:
            related_timestamps = get_scenario_timestamps(scenario_dict[track_uuid][related_uuid])
            removed_related:dict = deepcopy(scenario_dict[track_uuid])

            # I need a new data structure
            for track_uuid2, related_uuid2 in new_relationships:
                if track_uuid2 == track_uuid:
                    removed_related.pop(related_uuid2)

            if len(removed_related) == 0 or len(get_scenario_timestamps(removed_related)) == 0:
                removed_related = related_timestamps

            filtered_removed_related = scenario_at_timestamps(removed_related, related_timestamps)
            filtered_removed_related = {track_uuid : filtered_removed_related}

            if related_uuid not in reversed_scenario_dict:
                reversed_scenario_dict[related_uuid] = filtered_removed_related
            else:
                reversed_scenario_dict[related_uuid] = scenario_or([filtered_removed_related, reversed_scenario_dict[related_uuid]])

        return reversed_scenario_dict
    return wrapper


def get_scenario_timestamps(scenario_dict:dict) -> list:
    if not isinstance(scenario_dict, dict):
        #Scenario dict is a list of timestamps
        return scenario_dict

    timestamps = []
    for relationship in scenario_dict.values():
        timestamps.extend(get_scenario_timestamps(relationship))

    return sorted(list(set(timestamps)))


def get_scenario_uuids(scenario_dict:dict) -> list:
    if get_scenario_timestamps(scenario_dict):
        scenario_uuids = list(scenario_dict.keys())
        for child in scenario_dict.items():
            if isinstance(child, dict):
                scenario_uuids.extend(get_scenario_uuids(child))
        return list(set(scenario_uuids))
    else:
        return []


def reconstruct_track_dict(scenario_dict):
    track_dict = {}

    for track_uuid, related_objects in scenario_dict.items():
        if isinstance(related_objects, dict):
            timestamps = get_scenario_timestamps(related_objects)
            if len(timestamps) > 0:
                track_dict[track_uuid] = get_scenario_timestamps(related_objects)
        else:
            if len(related_objects) > 0:
                track_dict[track_uuid] = related_objects

    return track_dict


def reconstruct_relationship_dict(scenario_dict):
    #Reconstructing legacy relationship dict

    relationship_dict = {track_uuid: {} for track_uuid in scenario_dict.keys()}

    for track_uuid, child in scenario_dict.items():
        if not isinstance(child, dict):
            continue
        
        descendants = get_objects_and_timestamps(scenario_dict[track_uuid])
        for related_uuid, timestamps in descendants.items():
            relationship_dict[track_uuid][related_uuid] = timestamps

    return relationship_dict


def get_objects_and_timestamps(scenario_dict: dict) -> dict:
    track_dict = {}

    for uuid, related_children in scenario_dict.items():

        if isinstance(related_children, dict):
            track_dict[uuid] = get_scenario_timestamps(related_children)
            temp_dict = get_objects_and_timestamps(related_children)

            for child_uuid, timestamps in temp_dict.items():
                if child_uuid not in track_dict:
                    track_dict[child_uuid] = timestamps
                else:
                    track_dict[child_uuid] = sorted(list(set(track_dict[child_uuid] + list(timestamps))))
        else:
            if uuid not in track_dict:
                track_dict[uuid] = related_children
            else:
                track_dict[uuid] = sorted(list(set(track_dict[uuid] + list(related_children))))

    return track_dict


def print_indented_dict(d:dict, indent=0):
    """
    Recursively prints a dictionary with indentation.

    Args:
        d (dict): The dictionary to print.
        indent (int): The current indentation level (number of spaces).
    """
    for key, value in d.items():
        print(" " * indent + str(key) + ":")
        if isinstance(value, dict):
            print_indented_dict(value, indent=indent + 4)
        else:
            print(" " * (indent + 4) + str(value))


def output_scenario(
    scenario,
    description, 
    log_dir:Path, 
    output_dir, 
    is_gt=False,
    method_name='ref',
    visualize=False,
    **vis_kwargs):
    
    Path(output_dir/log_dir.name).mkdir(exist_ok=True)
    create_mining_pkl(description, scenario, log_dir, output_dir, method_name=method_name)

    if visualize:
        log_scenario_visualization_path = Path(output_dir/log_dir.name/'scenario visualizations')
        log_scenario_visualization_path.mkdir(exist_ok=True)

        for file in log_scenario_visualization_path.iterdir():
            if file.is_file() and file.stem.split(sep='_')[0] == description:
                file.unlink()

        visualize_scenario(scenario, log_dir, log_scenario_visualization_path, description=description, **vis_kwargs)


def get_related_objects(relationship_dict):
    track_dict = reconstruct_track_dict(relationship_dict)

    all_related_objects = {}

    for track_uuid, related_objects in relationship_dict.items():
        for related_uuid, timestamps in related_objects.items():
            if timestamps and related_uuid not in track_dict and related_uuid not in all_related_objects:
                all_related_objects[related_uuid] = timestamps
            elif timestamps and related_uuid not in track_dict and related_uuid in all_related_objects:
                all_related_objects[related_uuid] = sorted(set(all_related_objects[related_uuid]).union(timestamps))
            elif timestamps and related_uuid in track_dict and related_uuid not in all_related_objects:
                non_track_timestamps = set(track_dict[related_uuid]).difference(timestamps)
                if non_track_timestamps:
                    all_related_objects[related_uuid] = non_track_timestamps
            elif timestamps and related_uuid in track_dict and related_uuid in all_related_objects:
                non_track_timestamps = set(track_dict[related_uuid]).difference(timestamps)
                if non_track_timestamps:
                    all_related_objects[related_uuid] = sorted(set(all_related_objects[related_uuid]).union(non_track_timestamps))

    return all_related_objects


def create_mining_pkl(description, scenario, log_dir:Path, output_dir:Path, method_name='ref'):
    """
    Generates both a pkl file for evaluation and annotations for the scenario mining challenge.
    """

    #data_columns = ['log_id', 'prompt', 'track_uuid', 'mining_category', 'timestamp_ns']
    all_data = []
    frames = []

    log_id = log_dir.name
    (output_dir / log_id).mkdir(exist_ok=True)

    annotations = read_feather(log_dir / 'sm_annotations.feather')
    log_timestamps = np.sort(annotations['timestamp_ns'].unique())
    all_uuids = list(annotations['track_uuid'].unique())
    ego_poses = get_ego_SE3(log_dir)

    referred_objects = swap_keys_and_listed_values(reconstruct_track_dict(scenario))
    relationships = reconstruct_relationship_dict(scenario)
    related_objects = swap_keys_and_listed_values(get_related_objects(relationships))

    for timestamp in log_timestamps:
        frame = {}
        timestamp_annotations = annotations[annotations['timestamp_ns'] == timestamp]

        timestamp_uuids = list(timestamp_annotations['track_uuid'].unique())
        ego_to_city = ego_poses[timestamp]

        frame['seq_id'] = (log_id, description)
        frame['timestamp_ns'] = timestamp
        frame['ego_translation_m'] = list(ego_to_city.translation)
        frame['description'] = description

        n = len(timestamp_uuids)
        frame['translation_m'] = np.zeros((n, 3))
        frame['size'] = np.zeros((n,3), dtype=np.float32)
        frame['yaw'] = np.zeros(n, dtype=np.float32)
        frame['velocity_m_per_s'] = np.zeros((n,3))
        frame['label'] = np.zeros(n, dtype=np.int32)
        frame['name'] = np.zeros(n, dtype='<U31')
        frame['track_id'] = np.zeros(n, dtype=np.int32)
        frame['score'] = np.zeros(n, dtype=np.float32)

        for i, track_uuid in enumerate(timestamp_uuids):
            track_df = timestamp_annotations[timestamp_annotations['track_uuid'] == track_uuid]
            cuboid = CuboidList.from_dataframe(track_df)[0]
            
            if track_df.empty:
                continue

            ego_coords = track_df[['tx_m', 'ty_m', 'tz_m']].to_numpy()
            size = track_df[['length_m', 'width_m', 'height_m']].to_numpy()
            translation_m = ego_to_city.transform_from(ego_coords)
            yaw = Rotation.from_matrix(ego_to_city.compose(cuboid.dst_SE3_object).rotation).as_euler('zxy')[0]

            #if track_uuid in referred_uuids:
            if timestamp in referred_objects and track_uuid in referred_objects[timestamp]:
                category = "REFERRED_OBJECT"
                label = 0
            elif timestamp in related_objects and track_uuid in related_objects[timestamp]:
                category = "RELATED_OBJECT"
                label = 1
            else:
                category = "OTHER_OBJECT"
                label = 2

            frame['translation_m'][i,:] = translation_m
            frame['size'][i,:] = size
            frame['yaw'][i] = yaw
            frame['velocity_m_per_s'][i,:] = np.zeros(3)
            frame['label'][i] = label
            frame['name'][i] = category
            frame['track_id'][i] = all_uuids.index(track_uuid)

            if 'score' in track_df:
                frame['score'][i] = track_df['score']
            else:
                frame['score'][i] = 1

            all_data.append([log_id, description, track_uuid, category, timestamp])

        frames.append(frame)

    sequences = {(log_id, description): frames}
    save(sequences, output_dir / log_id / f'{description}_{log_id[:8]}_{method_name}_predictions.pkl')

    print(f'Scenario pkl file for {description}_{log_id[:8]} saved successfully.')

    return True


def referred_full_tracks(pkl_file_path):
    """
    Reconstructs a mining pkl file by propagating referred object labels across all instances
    of the same track_id and removing all other objects.
    
    Args:
        pkl_file_path: Path to the pkl file
    
    Returns:
        reconstructed_sequences: Dictionary containing the reconstructed sequences
    """
    import pickle
    
    # Load the pkl file
    with open(pkl_file_path, 'rb') as f:
        sequences = pickle.load(f)
    
    reconstructed_sequences = {}
    
    # Process each sequence
    for seq_name, frames in sequences.items():
        # First pass: identify all track_ids that were ever referred objects
        referred_track_ids = set()
        for frame in frames:
            mask = frame['label'] == 0  # 0 is for REFERRED_OBJECT
            referred_track_ids.update(frame['track_id'][mask])
        
        # Second pass: reconstruct frames
        new_frames = []
        for frame in frames:
            # Create mask for referred track_ids
            mask = np.isin(frame['track_id'], list(referred_track_ids))
            
            # Create new frame with only referred objects
            new_frame = {
                'seq_id': frame['seq_id'],
                'timestamp_ns': frame['timestamp_ns'],
                'ego_translation_m': frame['ego_translation_m'],
                'description': frame['description'],
                'translation_m': frame['translation_m'][mask],
                'size': frame['size'][mask],
                'yaw': frame['yaw'][mask],
                'velocity_m_per_s': frame['velocity_m_per_s'][mask],
                'label': np.zeros(mask.sum(), dtype=np.int32),  # All are referred objects
                'name': np.array(['REFERRED_OBJECT'] * mask.sum(), dtype='<U31'),
                'track_id': frame['track_id'][mask]
            }
            
            # If score exists in the original frame (for predictions), include it
            if 'score' in frame:
                new_frame['score'] = frame['score'][mask]
            
            new_frames.append(new_frame)
        
        reconstructed_sequences[seq_name] = new_frames
    
    return reconstructed_sequences    