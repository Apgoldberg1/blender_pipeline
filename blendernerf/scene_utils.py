import bpy
import math
import numpy as np

def make_camera(xyz: tuple = (5, 0, 5), rots: tuple = (45, 0, 90), FOV: int =120):

    new_cam_data = bpy.data.cameras.new(name="new_cam")
    new_cam_data.angle = math.radians(FOV)  #50 mm focal length

    new_cam = bpy.data.objects.new(name="new_cam", object_data = new_cam_data)

    cons = new_cam.constraints.new(type='TRACK_TO')
    cons.target = bpy.context.active_object

    new_cam.location = xyz
    new_cam.rotation_euler = (math.radians(rots[0]), 0, math.radians(rots[2]))
    bpy.context.collection.objects.link(new_cam)


    return new_cam

import bpy

def get_solid_material():
    solid_material = bpy.data.materials.new(name="SolidMaterial")
    solid_material.use_nodes = True
    solid_material.node_tree.nodes.clear()

    # Create a Diffuse BSDF node
    diffuse_node = solid_material.node_tree.nodes.new('ShaderNodeBsdfDiffuse')

    # Create a Material Output node
    output_node = solid_material.node_tree.nodes.new('ShaderNodeOutputMaterial')

    # Link the Diffuse BSDF node to the Material Output node
    solid_material.node_tree.links.new(diffuse_node.outputs['BSDF'], output_node.inputs['Surface'])

    return solid_material


def get_glass_material():
    glass_material = bpy.data.materials.new(name="GlassMaterial")
    glass_material.use_nodes = True
    glass_material.node_tree.nodes.clear()
    glass_node = glass_material.node_tree.nodes.new('ShaderNodeBsdfGlass')
    glass_node.inputs['Roughness'].default_value = .05
    output_node = glass_material.node_tree.nodes.new('ShaderNodeOutputMaterial')
    glass_material.node_tree.links.new(glass_node.outputs['BSDF'], output_node.inputs['Surface'])


    return glass_material


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

def add_background():
    world = bpy.context.scene.world

    world.use_nodes = True
    bg_node = world.node_tree.nodes.get('Background')
    if bg_node:
        bg_node.inputs['Color'].default_value = (.2, .2, .2, 1.0)


def add_background_noise():
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

def get_rot_matrix(angle, axis):
    cos, sin = math.cos, math.sin
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

    return rot_matrix
