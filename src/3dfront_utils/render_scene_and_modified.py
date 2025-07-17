import blenderproc as bproc
from collections import defaultdict
from typing import Dict, List, Union
import sys
import argparse
import os
from blenderproc.python.types.MaterialUtility import Material
from blenderproc.python.types.MeshObjectUtility import MeshObject
from blenderproc.python.utility.Utility import Utility
from blenderproc.python.writer.WriterUtility import WriterUtility
import h5py
import numpy as np
import random
from pathlib import Path
import json
import signal
from contextlib import contextmanager
import blenderproc.python.renderer.RendererUtility as RendererUtility
from time import time
import traceback
import bpy

# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("front_folder", help="Path to the 3D front file")
    parser.add_argument("future_folder", help="Path to the 3D Future Model folder.")
    parser.add_argument("front_3D_texture_folder", help="Path to the 3D FRONT texture folder.")
    parser.add_argument("front_json", help="Path to a 3D FRONT scene json file, e.g.6a0e73bc-d0c4-4a38-bfb6-e083ce05ebe9.json.")
    parser.add_argument("modified_front_json")
    parser.add_argument('cc_material_folder', nargs='?', default="resources/cctextures",
                        help="Path to CCTextures folder, see the /scripts for the download script.")
    parser.add_argument("output_folder", nargs='?', default="renderings",
                        help="Path to where the data should be saved")
    parser.add_argument("--n_views_per_scene", type=int, default=100,
                        help="The number of views to render in each scene.")
    parser.add_argument("--append_to_existing_output", type=bool, default=True,
                        help="If append new renderings to the existing ones.")
    parser.add_argument("--fov", type=int, default=90, help="Field of view of camera.")
    parser.add_argument("--res_x", type=int, default=224, help="Image width.")
    parser.add_argument("--res_y", type=int, default=224, help="Image height.")
    return parser.parse_args()

def create_camera_pose(camera_location, look_at_point, up_vector=np.array([0, 0, 1])):
    forward = (look_at_point - camera_location)
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, up_vector)
    right /= np.linalg.norm(right)

    up = np.cross(right, forward)

    rotation = np.eye(4)
    rotation[:3, 0] = right
    rotation[:3, 1] = up
    rotation[:3, 2] = -forward
    rotation[:3, 3] = camera_location

    return rotation

def assign_room(data: dict, index: int) -> str:
    smap = data["instance_segmaps"][index]
    ids, counts = np.unique(smap, return_counts=True)
    room_map = {k["idx"] : k["room_id"] for k in data["instance_attribute_maps"][index]}
    room_counter = {room: 0 for room in room_map.values()}
    for i, c in zip(ids, counts):
        room_counter[room_map[i]] += c
    return max(room_counter.items(), key=lambda x: x[1] if x[0] else -1000)[0]

def write_hdf5(output_dir_path: str, output_data_dict: Dict[str, List[Union[np.ndarray, list, dict]]],
               furniture_item_names: list[str],
               append_to_existing_output: bool = False, stereo_separate_keys: bool = False):
    """
    Saves the information provided inside of the output_data_dict into a .hdf5 container

    :param output_dir_path: The folder path in which the .hdf5 containers will be generated
    :param output_data_dict: The container, which keeps the different images, which should be saved to disc.
                             Each key will be saved as its own key in the .hdf5 container.
    :param append_to_existing_output: If this is True, the output_dir_path folder will be scanned for pre-existing
                                      .hdf5 containers and the numbering of the newly added containers, will start
                                      right where the last run left off.
    :param stereo_separate_keys: If this is True and the rendering was done in stereo mode, than the stereo images
                                 won't be saved in one tensor [2, img_x, img_y, channels], where the img[0] is the
                                 left image and img[1] the right. They will be saved in separate keys: for example
                                 for colors in colors_0 and colors_1.
    """

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    amount_of_frames = 0
    for data_block in output_data_dict.values():
        if isinstance(data_block, list):
            amount_of_frames = max([amount_of_frames, len(data_block)])

    # if append to existing output is turned on the existing folder is searched for the highest occurring
    # index, which is then used as starting point for this run
    if append_to_existing_output:
        frame_offset = 0
        # Look for hdf5 file with highest index
        for path in os.listdir(output_dir_path):
            if path.endswith(".hdf5"):
                index = path[:-len(".hdf5")]
                if index.isdigit():
                    frame_offset = max(frame_offset, int(index) + 1)
    else:
        frame_offset = 0

    if amount_of_frames != bpy.context.scene.frame_end - bpy.context.scene.frame_start:
        raise Exception("The amount of images stored in the output_data_dict does not correspond with the amount"
                        "of images specified by frame_start to frame_end.")

    for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end):

        # for each frame a new .hdf5 file is generated
        room = assign_room(output_data_dict, frame)
        furniture_item = furniture_item_names[frame]
        os.makedirs(os.path.join(output_dir_path, room, furniture_item), exist_ok=True)
        hdf5_path = os.path.join(output_dir_path, room, furniture_item, str(frame + frame_offset) + ".hdf5")
        with h5py.File(hdf5_path, "w") as file:
            # Go through all the output types
            print(f"Merging data for frame {frame} into {hdf5_path}")

            adjusted_frame = frame - bpy.context.scene.frame_start
            for key, data_block in output_data_dict.items():
                if adjusted_frame < len(data_block):
                    # get the current data block for the current frame
                    used_data_block = data_block[adjusted_frame]
                    if stereo_separate_keys and (bpy.context.scene.render.use_multiview or
                                                 used_data_block.shape[0] == 2):
                        # stereo mode was activated
                        WriterUtility._write_to_hdf_file(file, key + "_0", data_block[adjusted_frame][0])
                        WriterUtility._write_to_hdf_file(file, key + "_1", data_block[adjusted_frame][1])
                    else:
                        WriterUtility._write_to_hdf_file(file, key, data_block[adjusted_frame])
                else:
                    raise Exception(f"There are more frames {adjusted_frame} then there are blocks of information "
                                    f" {len(data_block)} in the given list for key {key}.")
            blender_proc_version = Utility.get_current_version()
            if blender_proc_version is not None:
                WriterUtility._write_to_hdf_file(file, "blender_proc_version", np.string_(blender_proc_version))


def render_scene(
    front_json: str, scene_output_folder: Path,
    materials: dict = {}, cam_Ts: list = [], targeted_furniture_items: list[str] = [], return_parameters: bool = False
):
    try:
        with time_limit(600): # per scene generation would not exceeds X seconds.
            start_time = time()

            bproc.init()
            RendererUtility.set_max_amount_of_samples(32)

            mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "blender_label_mapping.csv"))
            mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

            # set the light bounces
            bproc.renderer.set_light_bounces(diffuse_bounces=200, glossy_bounces=200, max_bounces=200,
                                             transmission_bounces=200, transparent_max_bounces=200)
            # set intrinsic parameters
            bproc.camera.set_intrinsics_from_blender_params(lens=args.fov / 180 * np.pi, image_width=args.res_x,
                                                            image_height=args.res_y,
                                                            lens_unit="FOV")

            cam_K = bproc.camera.get_intrinsics_as_K_matrix()

            # write camera intrinsics
            if not cam_intrinsic_path.exists():
                np.save(str(cam_intrinsic_path), cam_K)

            # read 3d future model info
            with open('submodules/BlenderProc-3DFront/examples/datasets/front_3d_with_improved_mat/model_info_revised.json', 'r') as f:
                model_info_data = json.load(f)
            model_id_to_label = {m["model_id"]: m["category"].lower().replace(" / ", "/") if m["category"] else 'others' for
                                 m in
                                 model_info_data}

            # load the front 3D objects
            loaded_objects = bproc.loader.load_front3d(
                json_path=str(front_json),
                future_model_path=str(future_folder),
                front_3D_texture_path=str(front_3D_texture_folder),
                label_mapping=mapping,
                model_id_to_label=model_id_to_label)

            categories_of_interest = ["bed", "sofa", "chair", "table"]
            mesh_positions = np.array([
                obj.get_location() for obj in loaded_objects 
                if any(cname in obj.get_name().lower() for cname in categories_of_interest)
            ])
            mesh_names = np.array([
                obj.get_name() for obj in loaded_objects 
                if any(cname in obj.get_name().lower() for cname in categories_of_interest)
            ])

            # -------------------------------------------------------------------------
            #          Sample materials
            # -------------------------------------------------------------------------
            cc_materials = bproc.loader.load_ccmaterials(args.cc_material_folder, ["Bricks", "Wood", "Carpet", "Tile", "Marble"])

            floors: List[MeshObject] = bproc.filter.by_attr(loaded_objects, "name", "Floor.*", regex=True)
            for floor in floors:
                # For each material of the object
                uid = floor.get_cp("uid")
                if uid in materials:
                    for i, m in enumerate(materials[uid]):
                        floor.set_material(i, cc_materials[m])
                else:
                    materials[uid] = [random.randint(0, len(cc_materials)-1) for _ in floor.get_materials()]
                    for i, m in enumerate(materials[uid]):
                        floor.set_material(i, cc_materials[m])

            baseboards_and_doors = bproc.filter.by_attr(loaded_objects, "name", "Baseboard.*|Door.*", regex=True)
            wood_floor_materials = bproc.filter.by_cp(cc_materials, "asset_name", "WoodFloor.*", regex=True)
            for obj in baseboards_and_doors:
                # For each material of the object
                uid = obj.get_cp("uid")
                if uid in materials:
                    for i, m in enumerate(materials[uid]):
                        obj.set_material(i, wood_floor_materials[m])
                else:
                    materials[uid] = [random.randint(0, len(wood_floor_materials)-1) for _ in obj.get_materials()]
                    for i, m in enumerate(materials[uid]):
                        obj.set_material(i, wood_floor_materials[m])

            walls = bproc.filter.by_attr(loaded_objects, "name", "Wall.*", regex=True)
            marble_materials = bproc.filter.by_cp(cc_materials, "asset_name", "Marble.*", regex=True)
            for wall in walls:
                # For each material of the object
                uid = wall.get_cp("uid")
                if uid in materials:
                    for i, m in enumerate(materials[uid]):
                        wall.set_material(i, marble_materials[m])
                else:
                    materials[uid] = [random.randint(0, len(marble_materials)-1) for _ in wall.get_materials()]
                    for i, m in enumerate(materials[uid]):
                        wall.set_material(i, marble_materials[m])

            # -------------------------------------------------------------------------
            #          Sample camera extrinsics
            # -------------------------------------------------------------------------
            # Init sampler for sampling locations inside the loaded front3D house
            point_sampler = bproc.sampler.Front3DPointInRoomSampler(loaded_objects)

            # Init bvh tree containing all mesh objects
            bvh_tree = bproc.object.create_bvh_tree_multi_objects(
                [o for o in loaded_objects if isinstance(o, bproc.types.MeshObject)])

            # filter some objects from the loaded objects, which are later used in calculating an interesting score
            interest_score_setting = {'ceiling': 0, 'column': 0, 'customizedpersonalizedmodel': 0, 'beam': 0, 'wallinner': 0,
                                      'slabside': 0, 'customizedfixedfurniture': 0, 'cabinet/lightband': 0, 'window': 0,
                                      'hole': 0, 'customizedplatform': 0, 'baseboard': 0, 'customizedbackgroundmodel': 0,
                                      'front': 0, 'walltop': 0, 'wallouter': 0, 'cornice': 0, 'sewerpipe': 0,
                                      'smartcustomizedceiling': 0, 'customizedfeaturewall': 0, 'customizedfurniture': 0,
                                      'slabtop': 0, 'baywindow': 0, 'door': 0, 'customized_wainscot': 0, 'slabbottom': 0,
                                      'back': 0, 'flue': 0, 'extrusioncustomizedceilingmodel': 0,
                                      'extrusioncustomizedbackgroundwall': 0, 'floor': 0, 'lightband': 0,
                                      'customizedceiling': 0, 'void': 0, 'pocket': 0, 'wallbottom': 0, 'chair': 10, 'sofa': 10,
                                      'table': 10, 'bed': 10}
            special_objects = []
            special_object_scores = {}
            for category_name, category_score in interest_score_setting.items():
                special_objects_per_category = [obj.get_cp("category_id") for obj in loaded_objects if check_name(obj.get_name(), category_name)]
                special_objects.extend(special_objects_per_category)
                unique_cat_ids = set(special_objects_per_category)
                for cat_id in unique_cat_ids:
                    special_object_scores[cat_id] = category_score

            # sample camera poses
            proximity_checks = {}
            floor_areas = np.array(point_sampler.get_floor_areas())
            cam_nums = np.ceil(floor_areas / floor_areas.sum() * n_cameras).astype(np.int16)
            n_tries = cam_nums * 3
            
            if cam_Ts:
                for cam in cam_Ts:
                    bproc.camera.add_camera_pose(cam)
            else:
                for floor_id, cam_num_per_scene in enumerate(cam_nums):
                    cam2world_matrices = []
                    coverage_scores = []
                    mesh_indices = []
                    tries = 0
                    while tries < n_tries[floor_id]:
                        # sample cam loc inside house
                        height = np.random.uniform(1.4, 1.8)
                        location = point_sampler.sample_by_floor_id(height, floor_id=floor_id)
                        index = np.argmin(np.linalg.norm(mesh_positions - location[None], axis=1))
                        closest_mesh_pos = mesh_positions[index]
                        # Sample rotation (fix around X and Y axis)
                        # rotation = np.random.uniform([1.2217, 0, 0], [1.338, 0, np.pi * 2])  # pitch, roll, yaw
                        # cam2world_matrix = bproc.math.build_transformation_mat(location, rotation)
                        cam2world_matrix = create_camera_pose(location, closest_mesh_pos)

                        # Check that obstacles are at least 1 meter away from the camera and have an average distance between 2.5 and 3.5
                        # meters and make sure that no background is visible, finally make sure the view is interesting enough
                        obstacle_check = bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, proximity_checks, bvh_tree)
                        coverage_score = bproc.camera.scene_coverage_score(cam2world_matrix, special_objects,
                                                                        special_objects_weight=special_object_scores)
                        # for sanity check
                        if obstacle_check and coverage_score >= 0.5:
                            cam2world_matrices.append(cam2world_matrix)
                            coverage_scores.append(coverage_score)
                            mesh_indices.append(index)
                            tries += 1
                    cam_ids = np.argsort(coverage_scores)[-cam_num_per_scene:]
                    for cam_id, cam2world_matrix in enumerate(cam2world_matrices):
                        if cam_id in cam_ids:
                            bproc.camera.add_camera_pose(cam2world_matrix)
                            cam_Ts.append(cam2world_matrix)
                            item_name: str = mesh_names[mesh_indices[cam_id]]
                            item_name = item_name.split(".")[0].replace("/", "").replace(" ", "")
                            targeted_furniture_items.append(item_name)

            # render the whole pipeline
            # bproc.renderer.enable_normals_output()
            data = bproc.renderer.render()
            default_values = {"location": [0, 0, 0], "cp_inst_mark": '', "cp_uid": '', "cp_jid": '', "cp_room_id": ""}
            data.update(bproc.renderer.render_segmap(
                map_by=["instance", "class", "cp_uid", "cp_jid", "cp_inst_mark", "cp_room_id", "location"],
                default_values=default_values))
            

            # write camera extrinsics
            data['cam_Ts'] = cam_Ts
            # write the data to a .hdf5 container
            write_hdf5(str(scene_output_folder), data,
                                    append_to_existing_output=args.append_to_existing_output, furniture_item_names=targeted_furniture_items)
            print('Time elapsed: %f.' % (time()-start_time))
            if return_parameters:
                return materials, cam_Ts, targeted_furniture_items

    except TimeoutException as e:
        print('Time is out: %s.' % scene_name)
        with open(failed_scene_name_file, 'a') as file:
            file.write(scene_name + "\n")
        sys.exit(0)
    except Exception as e:
        print(''.join(traceback.format_exception(type(e), e, e.__traceback__)))
        print('Failed scene name: %s.' % scene_name)
        with open(failed_scene_name_file, 'a') as file:
            file.write(scene_name + "\n")
        sys.exit(0)


class TimeoutException(Exception): pass
@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def get_folders(args):
    front_folder = Path(args.front_folder)
    future_folder = Path(args.future_folder)
    front_3D_texture_folder = Path(args.front_3D_texture_folder)
    cc_material_folder = Path(args.cc_material_folder)
    output_folder = Path(args.output_folder)
    if not output_folder.exists():
        output_folder.mkdir()
    return front_folder, future_folder, front_3D_texture_folder, cc_material_folder, output_folder


def check_name(name, category_name):
    return True if category_name in name.lower() else False


if __name__ == '__main__':
    '''Parse folders / file paths'''
    args = parse_args()
    front_folder, future_folder, front_3D_texture_folder, cc_material_folder, output_folder = get_folders(args)
    front_json = front_folder.joinpath(args.front_json)
    n_cameras = args.n_views_per_scene

    failed_scene_name_file = output_folder.parent.joinpath('failed_scene_names.txt')

    cam_intrinsic_path = output_folder.joinpath('cam_K.npy')

    if not front_folder.exists() or not future_folder.exists() \
            or not front_3D_texture_folder.exists() or not cc_material_folder.exists():
        raise Exception("One of these folders does not exist!")

    scene_name = front_json.name[:-len(front_json.suffix)]
    print('Processing scene name: %s.' % (scene_name))

    '''Pass those failure cases'''
    if failed_scene_name_file.is_file():
        with open(failed_scene_name_file, 'r') as file:
            failure_scenes = file.read().splitlines()
        if scene_name in failure_scenes:
            print('File in failure log: %s. Continue.' % (scene_name))
            sys.exit(0)

    '''Pass already generated scenes.'''
    scene_output_folder = output_folder.joinpath(scene_name)
    modified_scene_output_folder = Path(str(scene_output_folder) + "_modified")
    existing_n_renderings = 0

    if scene_output_folder.is_dir():
        existing_n_renderings = len(list(scene_output_folder.iterdir()))
        if existing_n_renderings >= n_cameras:
            print('Scene %s is already generated.' % (scene_output_folder.name))
            sys.exit(0)

    if args.append_to_existing_output:
        n_cameras = n_cameras - existing_n_renderings

    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    materials, cam_Ts, targeted_items = render_scene(front_json=front_json, scene_output_folder=scene_output_folder, return_parameters=True)
    render_scene(
        front_json=args.modified_front_json, scene_output_folder=modified_scene_output_folder, 
        targeted_furniture_items=targeted_items,
        materials=materials, cam_Ts=cam_Ts, return_parameters=False)