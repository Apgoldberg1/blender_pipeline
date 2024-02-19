import bpy
import pathlib
import math
import numpy as np
import multiprocessing
 
def make_camera(xyz: tuple = (5, 0, 5), rots: tuple = (45, 0, 90), FOV: int =39.6):

    new_cam_data = bpy.data.cameras.new(name="new_cam")
    new_cam_data.angle = math.radians(FOV)  #50 mm focal length
    new_cam = bpy.data.objects.new(name="new_cam", object_data = new_cam_data)

    cons = new_cam.constraints.new(type='TRACK_TO')
    cons.target = bpy.context.active_object

    new_cam.location = xyz
    new_cam.rotation_euler = (math.radians(rots[0]), 0, math.radians(rots[2]))
    bpy.context.collection.objects.link(new_cam)


    return new_cam

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

def get_glass_material():

    glass_material = bpy.data.materials.new(name="GlassMaterial")
    glass_material.use_nodes = True
    glass_material.node_tree.nodes.clear()
    glass_node = glass_material.node_tree.nodes.new('ShaderNodeBsdfGlass')
    output_node = glass_material.node_tree.nodes.new('ShaderNodeOutputMaterial')
    glass_material.node_tree.links.new(glass_node.outputs['BSDF'], output_node.inputs['Surface'])

    return glass_material

import bpy

def add_light(location=(5, 5, 8), energy=1000, color=(1, 1, 1), name="PointLight"):
    bpy.ops.object.light_add(type='POINT', location=location)

    light_obj = bpy.context.object

    # Set light properties
    light_obj.data.energy = energy
    light_obj.data.color = color
    light_obj.name = name

    return light_obj

def use_gpu(device_type):
    """WARNING THIS DOESN'T WORK"""

    # Enable the add-on
    bpy.ops.preferences.addon_enable(module="cycles")
    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    cycles_preferences.refresh_devices()

    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = device_type

    bpy.context.scene.cycles.device = 'GPU'
    cuda_devices = cycles_preferences.devices

    devices = cycles_preferences.devices

    devices[1]["use"] = 0
    print(devices[1]["name"], "deactivated")

    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        print("Device Name:", d["name"])
        print("Type:", d["type"])
        print("Use:", d["use"])
        print()

def get_min_z(obj):
    """
    Get the minimum Z-coordinate of the object's bounding box.
    """
    # Get the object's world matrix
    world_matrix = obj.matrix_world

    # Get the object's bounding box vertices in local coordinates
    local_bbox_verts = [v[:] for v in obj.bound_box]

    # Transform the bounding box vertices to world coordinates
    world_bbox_verts = [world_matrix @ v.co for v in obj.data.vertices]

    # Extract the Z-coordinates and find the minimum
    min_z = min(v.z for v in world_bbox_verts)

    return min_z

def add_background_blank():
    world = bpy.context.scene.world

    world.use_nodes = True
    bg_node = world.node_tree.nodes.get('Background')
    if bg_node:
        bg_node.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)

def add_background():
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree

    # Clear default nodes
    for node in tree.nodes:
        tree.nodes.remove(node)

    # Add render layer and composite nodes
    render_layers_node = tree.nodes.new('CompositorNodeRLayers')
    composite_node = tree.nodes.new('CompositorNodeComposite')

    # Add noise node
    noise_node = tree.nodes.new('CompositorNodeTexture')
    noise_node.texture = bpy.data.textures.new('Noise Texture', type='CLOUDS')  # You can change the noise type here
    noise_node.location = (600, 300)  # Adjust node location as needed

    noise_node.texture.intensity = 1.5

    # Add mix node
    mix_node = tree.nodes.new('CompositorNodeMixRGB')
    mix_node.blend_type = 'MULTIPLY'  # Adjust blend type as needed
    mix_node.location = (900, 300)

    # Connect nodes
    tree.links.new(render_layers_node.outputs['Image'], mix_node.inputs[1])
    tree.links.new(noise_node.outputs[0], mix_node.inputs[2])
    tree.links.new(mix_node.outputs[0], composite_node.inputs['Image'])




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

        for i in range(0, rot_res):
            rotz = 0
            for j in range(0, rot_res):
                render.filepath = f"{output_dir}/{stl_fname.stem}/{rotz:.2f}_{roty:.2f}"
                bpy.ops.render.render(write_still = True)

                rotate_camera(camera, camera_origin, 0, "Z")
                rotate_camera(camera, camera_origin, roty, "Y")
                rotate_camera(camera, camera.location, rotz, "Z")
                rotz += rotz_incr

            roty += roty_incr


        bpy.data.objects.remove(obj, do_unlink=True)



if __name__ == "__main__":
    mesh_dir = "./mesh_dir"
    image_path_name = "outputs"
    render_dir(mesh_dir, image_path_name, rot_res = 6)

