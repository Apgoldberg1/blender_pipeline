import bpy
import pathlib
import math
import numpy as np
import multiprocessing
from scene_utils import *
from get_camera_info import get_camera_extrinsics, get_camera_intrinsics, save_json, get_camera_extrinsics_log

def add_sphere_world():
    prev_active = bpy.context.view_layer.objects.active
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.mesh.primitive_uv_sphere_add(radius=50, location=(0, 0, 0))


    # Create a new material
    mat = bpy.data.materials.new(name="sphere_material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes

    # Remove the Principled BSDF node
    principled_node = nodes.get("Principled BSDF")
    if principled_node:
        nodes.remove(principled_node)

    # Add a Noise Texture node
    noise_node = nodes.new(type='ShaderNodeTexNoise')
    #noise_node = nodes.new(type='ShaderNodeTexBrick')
    #noise_node = nodes.new(type='ShaderNodeTexChecker')
    #noise_node.inputs['Color1'].default_value = (0, .8, 0, .8)
    #noise_node.inputs['Color2'].default_value = (.8, 0, 0, .8)

    #noise_node.inputs['Scale'].default_value = 2.4
    #noise_node.inputs['Color1'].default_value = (0, .8, 0, .8)
    #noise_node.inputs['Color2'].default_value = (.8, 0, 0, .8)
    #noise_node.inputs['Mortar Size'].default_value = 0

    # Adjust parameters of the Noise Texture node
    noise_node.inputs['Scale'].default_value = 4
    noise_node.inputs['Distortion'].default_value = 6
    noise_node.inputs['Detail'].default_value = 0
    noise_node.normalize = True

    # Create a material output node and link it
    output_node = nodes.get("Material Output")
    if output_node:
        links = mat.node_tree.links
        links.new(noise_node.outputs['Color'], output_node.inputs['Surface'])


    #sphere_material.use_backface_culling = True
    # Apply the material to the sphere
    bpy.context.object.data.materials.append(mat)

    bpy.context.view_layer.objects.active = prev_active

def rotate_camera(camera_obj, origin, angle: float, axis: str = 'Z'):
    cos = math.cos
    sin = math.sin

    rot_matrix = get_rot_matrix(angle, axis)

    camera_obj.location = np.dot(origin, rot_matrix)


def render_dir(mesh_dir: str, output_dir: str, rot_res: int = 4):
    bpy.context.scene.render.engine = 'CYCLES'

    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512

    bpy.context.scene.render.pixel_aspect_x = 1
    bpy.context.scene.render.pixel_aspect_y = 1
    bpy.context.scene.cycles.samples = 16

    use_gpu("CUDA")

    path_name = mesh_dir
    image_path_name = output_dir
     
    stl_root = pathlib.Path(path_name)
     
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()  #Delete defualt cube

    bpy.ops.object.select_all(action='DESELECT')
    render = bpy.context.scene.render

    #add_background_noise()
    #glass_material = get_glass_material()
    glass_material = get_solid_material()

    light = add_light()
    collection = bpy.data.collections.new("lights")
    bpy.context.scene.collection.children.link(collection)
    collection.objects.link(light)

    light2 = add_light(location=(-7, -5, 10), energy=2000, color=(1, 1, 1), name="PointLight")
    collection.objects.link(light2)
    bpy.ops.object.select_all(action='DESELECT')

    FOV = 90

     
    for stl_fname in stl_root.glob('**/*.stl'):
        bpy.ops.import_mesh.stl(filepath=str(stl_fname))
        obj = bpy.context.active_object
        obj.data.materials.append(glass_material)
        #camera.data.anlge is FOV in radians
        obj.display.show_shadows = False

        max_dim = max(obj.dimensions)
        obj.scale *= 7 / max_dim
        #scale_factor =  (2*(math.sqrt(30))*math.atan(math.radians(FOV)) / (max_dim))
        #obj.scale *= scale_factor


        #z_dim = obj.dimensions[2]
        #z_min = get_min_z(obj)

        # Try and move the object to the origin
        bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN')
        bpy.ops.object.location = (0, 0, 0)


        camera_origin = np.array([math.sqrt(30), 0, 0])
        camera = make_camera(xyz = camera_origin, rots = (90, 0, 90), FOV=FOV)
        bpy.context.scene.camera = camera

        bpy.ops.object.select_all(action='DESELECT')
        #bpy.ops.mesh.primitive_plane_add(size=100, enter_editmode=False, align='WORLD', location=(0, 0, -10))
        add_sphere_world()

        rotz_incr, roty_incr = (2 * math.pi) / rot_res, (2 * math.pi / 2) / rot_res
        roty, rotz = -rot_res / 2 * roty_incr, 0
        #roty, rotz = 0, 0

        scene = bpy.context.scene

        intrinsics_dict = get_camera_intrinsics(scene, camera)

        extrinsics_dicts = []


        for i in range(0, rot_res):
            rotz = 0
            for j in range(0, rot_res):
                render.filepath = f"{output_dir}/{stl_fname.stem}/{rotz:.2f}_{roty:.2f}"
                bpy.ops.render.render(write_still = True)

                rotate_camera(camera, camera_origin, 0, "Z")
                rotate_camera(camera, camera_origin, roty, "Y")
                rotate_camera(camera, camera.location, rotz, "Z")
                rotz += rotz_incr

                extrinsics_dicts.append(get_camera_extrinsics(scene, camera, f"{render.filepath}.png", mode='TRAIN'))
                #extrinsics_dicts.append(get_camera_extrinsics_log(f"{render.filepath}.png", (roty, rotz)))
                #log_extrinsics(scene, camera, f"{render.filepath}.png", f"{output_dir}/{stl_fname.stem}/", mode='TRAIN')

            roty += roty_incr

        intrinsics_dict["frames"] = extrinsics_dicts
        save_json(intrinsics_dict, f"{output_dir}/{stl_fname.stem}/transforms.json")


        bpy.data.objects.remove(obj, do_unlink=True)



if __name__ == "__main__":
    mesh_dir = "../mesh_dir"
    image_path_name = "outputs"
    render_dir(mesh_dir, image_path_name, rot_res = 20)

