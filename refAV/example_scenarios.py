from pathlib import Path
from av2.utils.io import read_feather
from utils import *
import paths


dataset_dir = paths.SM_DATA_DIR.parent / 'val'
output_dir = Path("output/sm_predictions/val")
log_id = '0b86f508-5df9-4a46-bc59-5b9536dbde9f'
log_dir = dataset_dir / log_id
annotations_df = read_feather(log_dir / 'sm_annotations.feather')
avm = get_map(log_dir)
all_uuids = annotations_df['track_uuid'].unique()

scenarios = [4,9,10,11,12,13]


#Secenario 1: vehicle in the wrong lane

if 1 in scenarios:
    description = 'vehicles in the wrong lane type'
    vehicles = get_objects_of_category(log_dir, category="VEHICLE")
    non_buses = scenario_not(is_category)(vehicles, log_dir, category="BUS")

    non_bus_in_bus_lane = on_lane_type(non_buses, log_dir, lane_type='BUS')
    non_bike_in_bike_lane = on_lane_type(non_buses, log_dir, lane_type='BIKE')

    in_wrong_lane_type = scenario_or([non_bus_in_bus_lane, non_bike_in_bike_lane])
    output_scenario(in_wrong_lane_type, description, log_dir, output_dir, relationship_edges=True)
    

#Scenario 2: changing_lane_to_left

if 2 in scenarios:
    
    description = 'vehicles changing lanes'
    vehicles = get_objects_of_category(log_dir, category='VEHICLE')
    lane_changes = changing_lanes(vehicles, log_dir)
    output_scenario(lane_changes, description, log_dir, output_dir,relationship_edges=True)


#Scenario 4: near_multiple_pedestrians
if 4 in scenarios:
    description='vehicle_near_multiple_pedestrians'
    vehicles = get_objects_of_category(log_dir, category='VEHICLE')
    peds = get_objects_of_category(log_dir, category='PEDESTRIAN')

    vehicles_near_peds = near_objects(vehicles, peds, log_dir, min_objects=2)
    output_scenario(vehicles_near_peds, description, log_dir, output_dir,relationship_edges=True)

    description='multiple pedestrians near a vehicle'
    vehicles = get_objects_of_category(log_dir, category='VEHICLE')
    peds = get_objects_of_category(log_dir, category='PEDESTRIAN')

    vehicles_near_peds = reverse_relationship(near_objects)(vehicles, peds, log_dir, min_objects=2)
    output_scenario(vehicles_near_peds, description, log_dir, output_dir,relationship_edges=True)


#Scenario 5: turning_left
if 5 in scenarios:

    description='turning left'

    vehicle_uuids = get_objects_of_category(log_dir, category='VEHICLE')
    left_turn = turning(vehicle_uuids, log_dir, direction='left')
    output_scenario(left_turn, description, log_dir, output_dir,relationship_edges=True)


#Scenario 6: waiting_for_pedestrian_to_cross

if 6 in scenarios:
    description='pedestrians crossing in front of vehicles'
    vehicles = get_objects_of_category(log_dir, category='VEHICLE')
    peds = get_objects_of_category(log_dir, category='PEDESTRIAN')

    peds = reverse_relationship(being_crossed_by)(vehicles, peds, log_dir)
    output_scenario(peds, description, log_dir, output_dir,relationship_edges=True)


#Scenario 7: at_stop_sign
if 7 in scenarios:

    description='active vehicle at stop sign'
    vehicles = get_objects_of_category(log_dir, category='VEHICLE')
    active_vehicles = scenario_not(stationary)(vehicles, log_dir)
    stopped_vehicles = at_stop_sign(active_vehicles, log_dir)

    output_scenario(stopped_vehicles, description, log_dir, output_dir,relationship_edges=True)


#Scenario 8: Pedestrians in drivable area not on crosswalk

if 8 in scenarios:
    description='jaywalking pedestrian'
    peds = get_objects_of_category(log_dir, category='PEDESTRIAN')
    peds_on_road = on_road(peds, log_dir)
    jaywalking_peds = scenario_not(at_pedestrian_crossing)(peds_on_road, log_dir)
    output_scenario(jaywalking_peds, description, log_dir, output_dir,relationship_edges=True)


#Scenario 9: The vehicle behind another vehicle being crossed by a jaywalking pedestrian
if 9 in scenarios:

    description = 'moving vehicle behind another vehicle being crossed by a jaywalking pedestrian'

    peds = get_objects_of_category(log_dir, category='PEDESTRIAN')
    peds_on_road = on_road(peds, log_dir)
    jaywalking_peds = scenario_not(at_pedestrian_crossing)(peds_on_road, log_dir)

    vehicles = get_objects_of_category(log_dir, category='VEHICLE')
    moving_vehicles = scenario_and([in_drivable_area(vehicles, log_dir), scenario_not(stationary)(vehicles, log_dir)])
    crossed_vehicles = being_crossed_by(moving_vehicles, jaywalking_peds, log_dir)
    behind_crossed_vehicle = get_objects_in_relative_direction(crossed_vehicles, moving_vehicles, log_dir,
                                                direction='backward', max_number=1, within_distance=25)

    output_scenario(behind_crossed_vehicle, description, log_dir, output_dir,relationship_edges=True)


#Scenario 10: Pedestrians walking between two stopped vehicles

if 10 in scenarios:

    description='pedestrian walking between two stopped vehicles'
    vehicles = get_objects_of_category(log_dir, category='VEHICLE')
    peds = get_objects_of_category(log_dir, category='PEDESTRIAN')

    stationary_vehicles = stationary(vehicles, log_dir)

    peds_behind = get_objects_in_relative_direction(stationary_vehicles, peds, log_dir, direction='behind', within_distance=5,lateral_thresh=.5)
    peds_in_front  = get_objects_in_relative_direction(stationary_vehicles, peds_behind, log_dir, direction='front', within_distance=5, lateral_thresh=.5)

    peds_beween_vehicles = scenario_and([peds_in_front, peds_in_front])
    output_scenario(peds_beween_vehicles, description, log_dir, output_dir,relationship_edges=True)

if 11 in scenarios:
    description = 'vehicle with another vehicle in their lane'
    vehicles = get_objects_of_category(log_dir, category='VEHICLE')
    vehicles_in_same_lane = in_same_lane(vehicles, vehicles, log_dir)
    visualize_scenario(vehicles_in_same_lane, log_dir, Path('.'), description=description)

if 12 in scenarios:
    description = 'vehicle being overtaken on their right'
    vehicles = get_objects_of_category(log_dir, category='VEHICLE')
    moving_vehicles = scenario_not(stationary)(vehicles, log_dir)
    overtaken_on_left = being_crossed_by(moving_vehicles, moving_vehicles, log_dir, direction='right', in_direciton='counterclockwise', forward_thresh=5, lateral_thresh=10)
    visualize_scenario(overtaken_on_left, log_dir, Path('.'), description=description, relationship_edges=True)


if 13 in scenarios:
    #Lane splitting is moving between two cars that are in adjacent lanes, usually during slow traffic
    description = 'lane splitting motorcycle'

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

    output_scenario(lane_splitting_motorcycles, description, log_dir, output_dir)
