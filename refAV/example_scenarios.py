from pathlib import Path
import numpy as np
from av2.utils.io import read_feather
from av2.map.map_api import ArgoverseStaticMap
from utils import *
import paths

#Nuscenes scenarios rule-based detection
#For each scenario, indicate timestamp and objects of interest

#dataset_dir = Path("/data3/shared/datasets/ArgoVerse2/Sensor/train")
#dataset_dir = Path("/home/crdavids/data/datasets/av2/sensor/train")
dataset_dir = paths.DEFAULT_DATA_DIR.parent / 'train'
output_dir = Path("/home/crdavids/Trinity-Sync/av2-api/output/pickles/train")
log_id = '00a6ffc1-6ce9-3bc3-a060-6006e9893a1a'
log_dir = dataset_dir / log_id
annotations_df = read_feather(log_dir / 'sm_annotations.feather')
avm = get_map(log_dir)
all_uuids = annotations_df['track_uuid'].unique()
is_gt = False

scenarios = [14,15,16,17]



#Secenario 1: accelerating_at_crosswalk

if 1 in scenarios:
    title = 'vehicles in the wrong lane type'
    vehicles = get_objects_of_category(log_dir, category="VEHICLE")
    non_buses = scenario_not(is_category)(vehicles, log_dir, category="BUS")

    non_bus_in_bus_lane = on_lane_type(non_buses, log_dir, lane_type='BUS')
    non_bike_in_bike_lane = on_lane_type(non_buses, log_dir, lane_type='BIKE')

    in_wrong_lane_type = scenario_or([non_bus_in_bus_lane, non_bike_in_bike_lane])
    output_scenario(in_wrong_lane_type, title, log_dir, output_dir, relationship_edges=True, is_gt=is_gt)
    

#Scenario 2: changing_lane_to_left

if 2 in scenarios:
    
    title = 'vehicles changing lanes'
    peds = get_objects_of_category(log_dir, "PEDESTRIAN")
    vehicles = get_objects_of_category(log_dir, category='VEHICLE')
    lane_changes = changing_lanes(vehicles, log_dir)
    output_scenario(lane_changes, title, log_dir, output_dir,relationship_edges=True, is_gt=is_gt)

#Scenario 3: high_lateral_acceleration

if 3 in scenarios:
    title = 'high lateral acceleration'
    accel_dict = scenario_not(has_lateral_acceleration)(all_uuids, log_dir, min_accel=-1, max_accel=1)
    output_scenario(accel_dict, title, log_dir, output_dir,relationship_edges=True, is_gt=is_gt)


#Scenario 4: near_multiple_pedestrians

if 4 in scenarios:
    title='multiple pedestrians near a vehicle'
    vehicles = get_objects_of_category(log_dir, category='VEHICLE')
    peds = get_objects_of_category(log_dir, category='PEDESTRIAN')

    vehicles_near_peds = near_objects(vehicles, peds, log_dir, min_objects=2)
    output_scenario(vehicles_near_peds, title, log_dir, output_dir,relationship_edges=True, is_gt=is_gt)


#Scenario 5: turning_left

if 5 in scenarios:

    title='turning left'

    vehicle_uuids = get_objects_of_category(log_dir, category='VEHICLE')
    left_turn = turning(vehicle_uuids, log_dir, direction='left')
    output_scenario(left_turn, title, log_dir, output_dir,relationship_edges=True, is_gt=is_gt)


#Scenario 6: waiting_for_pedestrian_to_cross

if 6 in scenarios:
    title='pedestrians crossing in front of vehicles'
    vehicles = get_objects_of_category(log_dir, category='VEHICLE')
    peds = get_objects_of_category(log_dir, category='PEDESTRIAN')
    stationary_vehicles = stationary(vehicles, log_dir)

    peds = reverse_relationship(being_crossed_by)(stationary_vehicles, peds, log_dir)
    output_scenario(peds, title, log_dir, output_dir,relationship_edges=True, is_gt=is_gt)


#Scenario 7: at_stop_sign

if 7 in scenarios:

    title='active vehicle at stop sign'
    vehicles = get_objects_of_category(log_dir, category='VEHICLE')
    active_vehicles = scenario_not(stationary)(vehicles, log_dir)
    stopped_vehicles = at_stop_sign(active_vehicles, log_dir)

    output_scenario(stopped_vehicles, title, log_dir, output_dir,relationship_edges=True, is_gt=is_gt)


#Scenario 8: Pedestrians in drivable area not on crosswalk

if 8 in scenarios:
    title='jaywalking pedestrian'
    peds = get_objects_of_category(log_dir, category='PEDESTRIAN')
    peds_on_road = on_road(peds, log_dir)
    jaywalking_peds = scenario_not(at_pedestrian_crossing)(peds_on_road, log_dir)
    output_scenario(jaywalking_peds, title, log_dir, output_dir,relationship_edges=True, is_gt=is_gt)


#Scenario 9: The vehicle behind another vehicle being crossed by a jaywalking pedestrian

if 9 in scenarios:

    title = 'the vehicle behind another vehicle being crossed by a jaywalking pedestrian'

    peds = get_objects_of_category(log_dir, category='PEDESTRIAN')
    peds_on_road = on_road(peds, log_dir)
    jaywalking_peds = scenario_not(at_pedestrian_crossing)(peds_on_road, log_dir)

    vehicles = get_objects_of_category(log_dir, category='VEHICLE')
    moving_vehicles = scenario_and([in_drivable_area(vehicles, log_dir), scenario_not(stationary)(vehicles, log_dir)])
    crossed_vehicles = being_crossed_by(moving_vehicles, jaywalking_peds, log_dir)
    behind_crossed_vehicle = get_objects_in_relative_direction(crossed_vehicles, moving_vehicles, log_dir,
                                                direction='backward', max_number=1, within_distance=25)

    output_scenario(behind_crossed_vehicle, title, log_dir, output_dir,relationship_edges=True, is_gt=is_gt)


#Scenario 10: Pedestrians walking between two stopped vehicles

if 10 in scenarios:

    title='pedestrian walking between two stopped vehicles'
    vehicles = get_objects_of_category(log_dir, category='VEHICLE')
    peds = get_objects_of_category(log_dir, category='PEDESTRIAN')

    stationary_vehicles = stationary(vehicles, log_dir)

    peds_behind = get_objects_in_relative_direction(stationary_vehicles, peds, log_dir, direction='behind', within_distance=5,lateral_thresh=.5)
    peds_in_front  = get_objects_in_relative_direction(stationary_vehicles, peds_behind, log_dir, direction='front', within_distance=5, lateral_thresh=.5)

    peds_beween_vehicles = scenario_and([peds_in_front, peds_in_front])
    output_scenario(peds_beween_vehicles, title, log_dir, output_dir,relationship_edges=True, is_gt=is_gt)


if 11 in scenarios:
    title = 'vehicle with another vehicle in their lane'
    vehicles = get_objects_of_category(log_dir, category='VEHICLE')
    vehicles_in_same_lane = in_same_lane(vehicles, vehicles, log_dir)
    visualize_scenario(vehicles_in_same_lane, log_dir, Path('.'), title=title)

if 12 in scenarios:
    title = 'vehicle being overtaken on their right'
    vehicles = get_objects_of_category(log_dir, category='VEHICLE')
    moving_vehicles = scenario_not(stationary)(vehicles, log_dir)
    overtaken_on_left = being_crossed_by(moving_vehicles, moving_vehicles, log_dir, direction='right', forward_thresh=5, lateral_thresh=10)
    visualize_scenario(overtaken_on_left, log_dir, Path('.'), title=title, relationship_edges=True, is_gt=is_gt)


if 13 in scenarios:
    #Lane splitting is moving between two cars that are in adjacent lanes, usually during slow traffic
    title = 'lane splitting motorcycle'

    #Getting motorcycles that are on the road and moving
    motorcycles = get_objects_of_category(log_dir, category='MOTORCYCLE')
    active_motocycles = scenario_not(stationary)(motorcycles, log_dir)

    #Getting vehicles that are to the left and right of any active motorcycle
    vehicles = get_objects_of_category(log_dir, category='VEHICLE')
    vehicles_left_of_motorcycle = get_objects_in_relative_direction(active_motocycles, vehicles, log_dir, direction='left', min_number=1, max_number=1, within_distance=4, lateral_thresh=2)
    vehicles_right_of_motorcycle = get_objects_in_relative_direction(active_motocycles, vehicles, log_dir, direction='right', min_number=1, max_number=1, within_distance=4, lateral_thresh=2)

    #Motorcycle must be in the same the same lane as one of the cars
    motorcycle_in_lane_to_left = in_same_lane(active_motocycles, vehicles_right_of_motorcycle, log_dir)
    motorcycle_in_lane_to_right = in_same_lane(active_motocycles, vehicles_left_of_motorcycle, log_dir)

    #The motorcycle can be in the same lane as either the car to the left or right of it
    lane_splitting_motorcycles = scenario_or([has_objects_in_relative_direction(motorcycle_in_lane_to_left, vehicles_left_of_motorcycle, log_dir, direction='left', within_distance=4, lateral_thresh=2),
                                              has_objects_in_relative_direction(motorcycle_in_lane_to_right, vehicles_right_of_motorcycle, log_dir, direction='right', within_distance=4, lateral_thresh=2)])

    output_scenario(lane_splitting_motorcycles, title, log_dir, output_dir)
