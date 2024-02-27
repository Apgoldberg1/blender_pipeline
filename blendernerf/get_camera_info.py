"""
based on blendernerf
"""
import os
import json
from scene_utils import get_rot_matrix
import numpy as np

def get_camera_intrinsics(scene, camera):
        camera_angle_x = camera.data.angle_x
        camera_angle_y = camera.data.angle_y

        # camera properties
        f_in_mm = camera.data.lens # focal length in mm
        scale = scene.render.resolution_percentage / 100
        width_res_in_px = scene.render.resolution_x * scale #width
        height_res_in_px = scene.render.resolution_y * scale # height
        optical_center_x = width_res_in_px / 2
        optical_center_y = height_res_in_px / 2

        # pixel aspect ratios
        size_x = scene.render.pixel_aspect_x * width_res_in_px
        size_y = scene.render.pixel_aspect_y * height_res_in_px
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

        #sensor fit and sensor size (and camera angle swap in specific cases)
        if camera.data.sensor_fit == 'AUTO':
            sensor_size_in_mm = camera.data.sensor_height if width_res_in_px < height_res_in_px else camera.data.sensor_width
            if width_res_in_px < height_res_in_px:
                sensor_fit = 'VERTICAL'
                camera_angle_x, camera_angle_y = camera_angle_y, camera_angle_x
            elif width_res_in_px > height_res_in_px:
                sensor_fit = 'HORIZONTAL'
            else:
                sensor_fit = 'VERTICAL' if size_x <= size_y else 'HORIZONTAL'

        else:
            sensor_fit = camera.data.sensor_fit
            if sensor_fit == 'VERTICAL':
                sensor_size_in_mm = camera.data.sensor_height if width_res_in_px <= height_res_in_px else camera.data.sensor_width
                if width_res_in_px <= height_res_in_px:
                    camera_angle_x, camera_angle_y = camera_angle_y, camera_angle_x

        # focal length for horizontal sensor fit
        if sensor_fit == 'HORIZONTAL':
            sensor_size_in_mm = camera.data.sensor_width
            s_u = f_in_mm / sensor_size_in_mm * width_res_in_px
            s_v = f_in_mm / sensor_size_in_mm * width_res_in_px * pixel_aspect_ratio

        # focal length for vertical sensor fit
        if sensor_fit == 'VERTICAL':
            s_u = f_in_mm / sensor_size_in_mm * width_res_in_px / pixel_aspect_ratio
            s_v = f_in_mm / sensor_size_in_mm * width_res_in_px

        camera_intr_dict = {
            'camera_angle_x': camera_angle_x,
            'camera_angle_y': camera_angle_y,
            'fl_x': s_u,
            'fl_y': s_v,
            'k1': 0.0,
            'k2': 0.0,
            'p1': 0.0,
            'p2': 0.0,
            'cx': optical_center_x,
            'cy': optical_center_y,
            'w': width_res_in_px,
            'h': height_res_in_px,
            #'aabb_scale': scene.aabb
        }
        return camera_intr_dict

def get_camera_extrinsics(scene, camera, name, mode='TRAIN'):
    assert mode == 'TRAIN' or mode == 'TEST'

    filename = os.path.basename(name)
    #filedir =  * (mode == 'TRAIN') + OUTPUT_TEST * (mode == 'TEST')

    frame_data = {
        'file_path': os.path.join("", filename),
        'transform_matrix': listify_matrix(camera.matrix_world)
    }

    return frame_data

def add_to_json_file(filename, data, wipe=False):
    if os.path.exists(filename) and not wipe:
        with open(filename, 'r+') as file:
            # Load existing JSON data
            json_data = json.load(file)
            # Append new data to the JSON
            json_data.append(data)
            # Move the file cursor to the beginning to overwrite the file
            file.seek(0)
            # Write the updated JSON data
            json.dump(json_data, file, indent=4)
            file.truncate()  # Truncate any remaining content if the new data is smaller
    else:
        if os.path.exists(filename):
            os.remove(filename)
        with open(filename, 'w') as file:
            json.dump([data], file, indent=4)

def save_json(data, filename):
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

def save_extrinsics(extr_dict, output_dir):
    add_to_json_file(os.path.join(output_dir, "transforms.json"), extr_dict)


def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

