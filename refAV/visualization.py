from pathlib import Path
import pyvista as pv
from copy import deepcopy
from av2.structures.cuboid import Cuboid, CuboidList
from av2.geometry.se3 import SE3
import vtk
import numpy as np
import matplotlib
from av2.map.lane_segment import LaneSegment
import av2.geometry.polyline_utils as polyline_utils
import av2.rendering.vector as vector_plotting_utils
from av2.structures.sweep import Sweep
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.map.map_api import ArgoverseStaticMap
import matplotlib.pyplot as plt
from av2.datasets.sensor.sensor_dataloader import SensorDataloader
from av2.datasets.sensor.constants import RingCameras
import json
from av2.rendering.video import tile_cameras
import pandas as pd
from scipy.spatial.transform import Rotation as R
import cv2

from refAV.dataset_conversion import pickle_to_feather
from refAV.paths import AV2_DATA_DIR
from refAV.utils import (
    get_map, reconstruct_relationship_dict, reconstruct_track_dict,
    swap_keys_and_listed_values, get_log_split, read_feather,
    get_ego_uuid, get_nth_pos_deriv, dict_empty, get_ego_SE3
)


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
                font_family='times'
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
                        with_map=True, with_cf=False, with_lidar=True, relationship_edges=False, stride=1, av2_log_dir=None,
                        display_progress=True, save_pdfs=False):
    """
    Generate a birds-eye-view video of a series of LiDAR scans.
    
    :param lidar_files: List of file paths to LiDAR scan data.
    :param output_file: Path to the output video file.
    """

    #Conversion to legacy code
    scenario_dict = reconstruct_track_dict(scenario)
    relationship_dict = reconstruct_relationship_dict(scenario)

    FPS = 10
    output_file = output_dir / (description + '.mp4')
    plotter = pv.Plotter(off_screen=True)
    plotter.open_movie(output_file, framerate=FPS)

    if av2_log_dir is None:
        split = get_log_split(log_dir)
        av2_log_dir = AV2_DATA_DIR / split / log_dir.name

    dataset = AV2SensorDataLoader(data_dir=av2_log_dir.parent, labels_dir=av2_log_dir.parent)
    log_id = log_dir.name

    set_camera_position_pv(plotter, scenario_dict, relationship_dict, log_dir)
    #plotter.add_legend([(description,'black'),(log_dir.name,'black')],
    #                    bcolor='white', border=True, loc='upper left',size=(.7,.1))

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
    frequency = 1/(float(timestamps[1] - timestamps[0])/1E9)

    for i in range(0, len(timestamps), stride):
        if display_progress:
            print(f'{i}/{len(timestamps)}', end='\r')

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
    
        if save_pdfs:
            plotter.save_graphic(output_dir / f'{description}_{timestamp}.pdf')

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
    print(scenario_height)
    camera_height = min(max(scenario_height+height_above_scenario-20, scenario_height), scenario_height+100)
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
            bcolor='white', border=True, loc='upper left',size=(.267,.13), font_family='times')
    
    for j in range(3):
        plotter.write_frame()

    plotter.add_legend([(' referred objects','lime', 'rectangle'),(' related objects','blue', 'rectangle'), (' other objects', 'red', 'rectangle'), (' 2', 'orange')],
            bcolor='white', border=True, loc='upper left',size=(.267,.13),font_family='times')
    
    for j in range(3):
        plotter.write_frame()        

    plotter.add_legend([(' referred objects','lime', 'rectangle'),(' related objects','blue', 'rectangle'), (' other objects', 'red', 'rectangle'), ('  1', 'yellow')],
            bcolor='white', border=True, loc='upper left',size=(.267,.13),font_family='times')
    
    for j in range(3):
        plotter.write_frame()
    plotter.add_legend([(' referred objects','lime', 'rectangle'),(' related objects','blue', 'rectangle'), (' other objects', 'red', 'rectangle'), ('   GO!', 'green')],
            bcolor='white', border=True, loc='upper left',size=(.267,.13),font_family='times')
    
    for j in range(3):
        plotter.write_frame()

    plotter.add_legend([(' referred objects','lime', 'rectangle'),(' related objects','blue', 'rectangle'), (' other objects', 'red', 'rectangle')],
        bcolor='white', border=True, loc='upper left',size=(.2,.1),font_family='times')


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
                        bcolor='white', border=True, loc='upper left',size=(.7,.1),font_family='times')

    for i in range(4):
        actors = []

        actors.extend(plot_cuboids(scenario_cuboids, plotter, scenario_transforms, color='lime', opacity=1))
        actors.extend(plot_cuboids(related_cuboids, plotter, related_transforms, color='blue', opacity=1))

        for j in range(5):
            plotter.write_frame()

        plotter.remove_actor(actors)

        for j in range(5):
            plotter.write_frame()

"""
def visualize_rgb(
    dataset_dir: Path,
    cuboid_csv: Path,
    output_vis_dir: Path,
    log_id: str,
    description: str,
    annotation_dir = Path('/data3/crdavids/refAV/dataset/test/'),
    ) -> None:
    
    Generates videos visualizing the 3D bounding boxes from an output feather file.
    Only used for visualizing the ground truth
    
    print(f"Generating visualization for {log_id} - {description}...")

    if not cuboid_csv.exists():
        print(f"Error: Cuboid CSV not found: {cuboid_csv}")
        return

    # --- Data Loading ---
    # Determine camera names (using RingCameras as default)
    valid_ring = {x.value for x in RingCameras}
    cam_names_str = tuple(x.value for x in RingCameras)
    cam_enums = [RingCameras(cam) for cam in cam_names_str if cam in valid_ring]
    cam_names=tuple(cam_enums)

    # Load AV2 dataloader for the specific log
    try:
        dataloader = SensorDataloader(
            dataset_dir, # Should be the root AV2 'Sensor' directory
            with_annotations=False, # We load annotations from our CSV
        )
    except Exception as e:
        print(f"Error initializing SensorDataloader for log {log_id}: {e}")
        return

    # Load cuboids from the generated CSV
    try:
        df = pd.read_feather(cuboid_csv)
        if df.empty:
            print(f"No cuboids found in {cuboid_csv}. Skipping visualization.")
            return
    except Exception as e:
        print(f"Error loading or parsing cuboid CSV {cuboid_csv}: {e}")
        return

    # --- Rendering Loop ---
    output_log_dir = output_vis_dir / f"{log_id}_{description}" # Shorten desc for filename
    output_log_dir.mkdir(parents=True, exist_ok=True)
    rendered_count = 0

    with open('baselines/groundingSAM/log_id_to_start_index.json', 'rb') as file:
        log_id_to_start_index = json.load(file)

    i = log_id_to_start_index[log_id]
    datum = dataloader[i]
    annotations_path = annotation_dir / log_id / 'sm_annotations.feather'
    anno_df = pd.read_feather(annotations_path)

    while datum.log_id == log_id:

        try:
            i += 2
            datum = dataloader[i]
            timestamp = datum.timestamp_ns
            timestamp_df = df[df['timestamp_ns'] == timestamp]
            referred_df = timestamp_df[timestamp_df['mining_category'] == 'REFERRED_OBJECT']

            anno_timestamp_df = anno_df[anno_df['timestamp_ns'] == timestamp]
            anno_cuboids = CuboidList.from_dataframe(anno_timestamp_df)
            plotted_uuids = referred_df['track_uuid'].unique()
        

            # Filter cuboids for the current timestamp
            ts_cuboids = [c for c in anno_cuboids.cuboids if c.track_uuid in plotted_uuids]
            if not ts_cuboids:
                continue # Skip frames with no detections

            current_cuboid_list = CuboidList(ts_cuboids)
            timestamp_city_SE3_ego_dict = datum.timestamp_city_SE3_ego_dict
            synchronized_imagery = datum.synchronized_imagery

            if synchronized_imagery is not None:
                cam_name_to_img = {}
                for cam_name, cam in synchronized_imagery.items():
                    if cam.timestamp_ns in timestamp_city_SE3_ego_dict:
                        city_SE3_ego_cam_t = timestamp_city_SE3_ego_dict[cam.timestamp_ns]
                        img = cam.img.copy()
                        img = current_cuboid_list.project_to_cam(
                            img,
                            cam.camera_model,
                            city_SE3_ego_cam_t,
                            city_SE3_ego_cam_t,
                        )
                        cam_name_to_img[cam_name] = img
                                        # Save the frame

                        out_path = output_log_dir / cam_name / f"{description}_{timestamp}.png"
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(out_path), img)
            
                if len(cam_name_to_img) < len(cam_names):
                    continue


        except Exception as e:
            print(f"\nError during rendering frame {i} for log {log_id}: {e}")
            import traceback
            traceback.print_exc()
            continue # Try next frame

    print(f"Finished visualization for {log_id} - {description}. Frames saved in {output_log_dir}")

    
def plot_feather_heatmap(df):


    # Read the Feather file

    #df = pd.read_feather(df)
    # Extract the data
    referred_df = df[df['mining_category'].isin(['REFERRED_OBJECT'])]
    x = referred_df['tx_m']
    y = referred_df['ty_m']

    # Create the heatmap using hist2d
    # You can adjust 'bins' to change the resolution of your heatmap
    plt.rcParams['font.family'] = 'serif'

    hist_all, xedges, yedges = np.histogram2d(y, x, bins=50, range=[(-200, 200), (-200, 200)])

    fig, ax = plt.subplots()
    im = ax.imshow(hist_all.T , origin='lower',
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
               cmap='viridis', aspect='auto', norm='log') # Using 'coolwarm' for difference visualization
    plt.colorbar(im, ax=ax, label='number of bounding boxes') # Updated label for normalized difference
    plt.ylabel('x-position (meters)')
    plt.xlabel('y-position (meters)')
    ax.set_xlim((-200, 200))
    ax.set_ylim((-200, 200))
    ax.set_title('RefAV Referred Object Distribution')
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig('output/heatmap4.pdf')

def plot_distribution_difference_heatmap(df):

    df_referred_objects = df[df['mining_category'].isin(['REFERRED_OBJECT', 'RELATED_OBJECT'])]
    df_all_objects = df[df['mining_category'].isin(['REFERRED_OBJECT', 'RELATED_OBJECT', 'OTHER_OBJECT'])]
    # Extract relevant data from df_all_objects
    x_all = df_all_objects['tx_m']
    y_all = df_all_objects['ty_m']

    # Extract relevant data from df_referred_objects
    x_referred = df_referred_objects['tx_m'] + 1e-9
    y_referred = df_referred_objects['ty_m'] + 1e-9

    # Define common bins and ranges for consistent comparison
    bins = 100
    x_range = (-200, 200)
    y_range = (-200, 200)

    # Create 2D histograms for both dataframes, normalized by the number of observations
    # Using 'density=True' to normalize the histograms such that the sum over all bins equals 1.
    hist_all, xedges, yedges = np.histogram2d(y_all, x_all, bins=bins, range=[y_range, x_range], density=True)
    hist_referred, _, _ = np.histogram2d(y_referred, x_referred, bins=bins, range=[y_range, x_range], density=True)

    # Calculate the difference in normalized distributions
    difference_hist = 100*hist_referred / (hist_all)
    difference_hist = np.clip(np.nan_to_num(difference_hist), 0, 300)
    print(difference_hist.shape)
    norm = TwoSlopeNorm(vmin=0, vcenter=100, vmax=300)

    # Plot the difference heatmap
    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots() # Get the axes object

    im = ax.imshow(difference_hist.T , origin='lower',
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
               cmap='RdYlGn', aspect='auto', norm=norm) # Using 'coolwarm' for difference visualization
    plt.colorbar(im, ax=ax, label='Change from AV2 Sensor Distribution (%)') # Updated label for normalized difference
    plt.ylabel('x-position (meters)')
    plt.xlabel('y-position (meters)')
    ax.set_xlim((-52, 52))
    ax.set_ylim((-52, 52))
    ax.set_title('RefAV vs AV2 Sensor Object Distribution')
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig('output/normalized_distribution_difference_heatmap.pdf')
    

if __name__ == "__main__":
    
    import refAV.paths as paths

    dataset_dir = paths.AV2_DATA_DIR
    feather_path = Path('/home/crdavids/Trinity-Sync/refbot/output/visualization/b40c0cbf-5d35-30df-9f63-de088ada278e/turning left_b40c0cbf_annotations.feather') 
    output_dir = Path('output/visualization')
    log_id = 'b40c0cbf-5d35-30df-9f63-de088ada278e'

    visualize_rgb(dataset_dir, feather_path, output_dir, log_id, description="Vehicle making left turn through ego-vehicle's path while it is raining")

    pass
"""