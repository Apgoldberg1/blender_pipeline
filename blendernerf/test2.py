import bpy
import pathlib
import math
import numpy as np
import multiprocessing
from scene_utils import *
from get_camera_info import get_camera_extrinsics, get_camera_intrinsics, save_json

def rotate_camera(camera_obj, origin, angle: float, axis: str = 'Z'):
    cos = math.cos
    sin = math.sin

    if axis == 'Z':
        rot_matrix = np.array([
            [cos(angle), -sin(angle), 0],
            [sin(angle), cos(angle), 0],
            [0, 0, 1]
            ])
    elif axis == 'Y':
        rot_matrix = np.array([
            [cos(angle), 0, sin(angle)],
            [0, 1, 0],
            [-sin(angle), 0, -cos(angle)]
            ])
    elif axis == 'X':
        rot_matrix = np.array([
            [1, 0, 0],
            [0, cos(angle), -sin(angle)],
            [0, sin(angle), cos(angle)]
            ])
    else:
        raise AssertionError("axis must be 'X', 'Y', or 'Z'")

    camera_obj.location = np.dot(origin, rot_matrix)


def render_dir(mesh_dir: str, output_dir: str, rot_res: int = 4):
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 256

    use_gpu("CUDA")

    path_name = mesh_dir
    image_path_name = output_dir
     
    stl_root = pathlib.Path(path_name)
     
    bpy.ops.object.delete()  #Delete defualt cube

    bpy.ops.object.select_all(action='DESELECT')
    render = bpy.context.scene.render

    add_background()
    glass_material = get_glass_material()

    light = add_light()
    collection = bpy.data.collections.new("lights")
    bpy.context.scene.collection.children.link(collection)
    collection.objects.link(light)

    light2 = add_light(location=(-7, -5, 10), energy=2000, color=(1, 1, 1), name="PointLight")
    collection.objects.link(light2)

     
    for stl_fname in stl_root.glob('**/*.stl'):
        bpy.ops.import_mesh.stl(filepath=str(stl_fname))
        obj = bpy.context.active_object
        obj.data.materials.append(glass_material)
        #camera.data.anlge is FOV in radians
        obj.display.show_shadows = False

        max_dim = max(obj.dimensions)
        scale_factor =  .15 * (2*math.sqrt(50)*math.atan(.69)) / (max_dim)
        obj.scale *= scale_factor

        #z_dim = obj.dimensions[2]
        #z_min = get_min_z(obj)

        # Try and move the object to the origin
        bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN')
        bpy.ops.object.location = (0, 0, 0)


        camera_origin = np.array([math.sqrt(50), 0, 0])
        camera = make_camera(xyz = camera_origin, rots = (90, 0, 90))
        bpy.context.scene.camera = camera

        bpy.ops.object.select_all(action='DESELECT')
        #bpy.ops.mesh.primitive_plane_add(size=100, enter_editmode=False, align='WORLD', location=(0, 0, -10))

        #bpy.context.view_layer.objects.active = camera
        #camera.select_set(True)
        #bpy.ops.object.origin_set(type="ORIGIN_CURSOR")

        rotz_incr, roty_incr = (2 * math.pi) / rot_res, (math.pi / 2) / rot_res
        roty, rotz = 0, 0

        scene = bpy.context.scene
        #log_intrinsics(scene, camera, f"{output_dir}/{stl_fname.stem}/")
        intrinsics_dict = get_camera_intrinsics(scene, camera)

        extrinsics_dicts = []


        for i in range(0, rot_res):
            rotz = 0
            for j in range(0, rot_res):
                render.filepath = f"{output_dir}/{stl_fname.stem}/{rotz:.2f}_{roty:.2f}"
                #bpy.ops.render.render(write_still = True)

                rotate_camera(camera, camera_origin, 0, "Z")
                rotate_camera(camera, camera_origin, roty, "Y")
                rotate_camera(camera, camera.location, rotz, "Z")
                rotz += rotz_incr

                extrinsics_dicts.append(get_camera_extrinsics(scene, camera, f"{render.filepath}.png", mode='TRAIN'))
                #log_extrinsics(scene, camera, f"{render.filepath}.png", f"{output_dir}/{stl_fname.stem}/", mode='TRAIN')

            roty += roty_incr

        intrinsics_dict["frames"] = extrinsics_dicts
        save_json(intrinsics_dict, f"{output_dir}/{stl_fname.stem}/transforms.json")


        bpy.data.objects.remove(obj, do_unlink=True)



if __name__ == "__main__":
    mesh_dir = "../mesh_dir"
    image_path_name = "outputs"
    render_dir(mesh_dir, image_path_name, rot_res = 7)

