import bpy
import math
import mathutils
from mathutils import Matrix, Vector
import bmesh
import numpy as np
import os
import json
from random import random, uniform, choice

##################################################################
#                       USER PARAMETERS                          #
##################################################################

# Number of sequences to generate
num_sequences = 1000

# Number of frames per sequence (physics simulation length)
num_frames = 100

# Frame rate (fps) for the simulation
fps = 25

# Cameras
num_static_cams = 25         # Number of cameras used for t=0
num_dynamic_cams = 5         # Subset used for t>0
camera_radius = 35           # Radius of the hemisphere/circle on which cameras lie

# Output folder where all sequences go
base_output_folder = f"/Users/williambittner/Documents/Blender/datasets/bouncing_spheres_{num_sequences}seq_{num_frames}ts"

# Floor parameters
floor_size = 20.0          # X-Y dimension of the floor
floor_thickness = 1      # thickness
floor_location_z = -1.0    # Shift the floor downward

# Spheres
num_spheres = 5
sphere_min_radius = 0.4
sphere_max_radius = 0.8
max_initial_z = 8.0         # max random height to place the spheres

# Rigid body physics settings
gravity = -9.81
bounce_restitution = 0.85  # higher bounciness
bounce_friction = 0.0
collision_margin = 0.00001     # use a standard collision margin

# Number of points to sample for the floor + spheres point cloud
n_points_per_mesh = 1500

##################################################################
#               HELPER FUNCTIONS AND GLOBALS                     #
##################################################################

def ensure_dir(path):
    """Ensure a directory exists."""
    os.makedirs(path, exist_ok=True)

def make_serializable(matrix):
    """Convert a Blender Matrix into a nested Python list for JSON."""
    return [list(row) for row in matrix]

def random_point_in_triangle(v1, v2, v3):
    """Generate a random point inside a triangle (v1,v2,v3 are BMVerts)."""
    r1, r2 = random(), random()
    sqrt_r1 = math.sqrt(r1)
    u = 1 - sqrt_r1
    v = sqrt_r1 * (1 - r2)
    w = r2 * sqrt_r1
    return (u * v1.co) + (v * v2.co) + (w * v3.co)

def random_point_in_polygon(verts):
    """Generate a random point inside a polygon by triangulating from vert[0]."""
    triangles = [(verts[0], verts[i], verts[i + 1]) for i in range(1, len(verts) - 1)]
    chosen_triangle = choice(triangles)
    return random_point_in_triangle(*chosen_triangle)

def setup_render_settings(res_x=600, res_y=600, samples=32, engine='BLENDER_EEVEE'):
    """Configure basic render settings."""
    scene = bpy.context.scene
    scene.render.engine = engine
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y
    # If using Cycles, set scene.cycles.samples = samples
    scene.eevee.taa_samples = samples

##################################################################
#                    CREATE EMISSIVE MATERIAL                    #
##################################################################

def create_emissive_color_grid_material(mat_name="EmissiveColorGridMat"):
    """
    Create a material that uses an image texture with a generated Color Grid pattern,
    connected to an Emission shader for self-illumination.
    """
    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear all nodes.
    for node in nodes:
        nodes.remove(node)

    # Create a Texture Coordinate node.
    tex_coord_node = nodes.new(type='ShaderNodeTexCoord')
    tex_coord_node.location = (-600, 0)

    # Create an Image Texture node with a generated Color Grid image.
    image_tex_node = nodes.new(type='ShaderNodeTexImage')
    image_tex_node.location = (-400, 0)
    img = bpy.data.images.new("ColorGridImage", width=512, height=512)
    img.generated_type = 'COLOR_GRID'
    image_tex_node.image = img

    # Create an Emission node.
    emission_node = nodes.new(type='ShaderNodeEmission')
    emission_node.location = (-100, 0)
    emission_node.inputs["Strength"].default_value = 5.0

    # Create the Material Output node.
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    output_node.location = (200, 0)

    # Connect the nodes.
    links.new(tex_coord_node.outputs["UV"], image_tex_node.inputs["Vector"])
    links.new(image_tex_node.outputs["Color"], emission_node.inputs["Color"])
    links.new(emission_node.outputs["Emission"], output_node.inputs["Surface"])

    return mat


def create_emissive_color_grid_material_floor(mat_name="EmissiveColorGridMat"):
    """
    Create a material that uses an image texture with a generated Color Grid pattern,
    connected to an Emission shader for self-illumination.
    """
    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear all nodes.
    for node in nodes:
        nodes.remove(node)

    # Create a Texture Coordinate node.
    tex_coord_node = nodes.new(type='ShaderNodeTexCoord')
    tex_coord_node.location = (-600, 0)
    
    # Create a Mapping node to scale the coordinates.
    mapping_node = nodes.new(type='ShaderNodeMapping')
    mapping_node.location = (-600, 0)
    # Set a scaling factor (adjust these values as needed)
    mapping_node.inputs["Scale"].default_value = (0.1, 0.1, 0.1)

    # Create an Image Texture node with a generated Color Grid image.
    image_tex_node = nodes.new(type='ShaderNodeTexImage')
    image_tex_node.location = (-400, 0)
    img = bpy.data.images.new("ColorGridImage", width=512, height=512)
    img.generated_type = 'COLOR_GRID'
    image_tex_node.image = img

    # Create an Emission node.
    emission_node = nodes.new(type='ShaderNodeEmission')
    emission_node.location = (-100, 0)
    emission_node.inputs["Strength"].default_value = 5.0

    # Create the Material Output node.
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    output_node.location = (200, 0)

    # Connect the nodes.
    links.new(tex_coord_node.outputs["Object"], mapping_node.inputs["Vector"])
    links.new(mapping_node.outputs["Vector"], image_tex_node.inputs["Vector"])
    links.new(image_tex_node.outputs["Color"], emission_node.inputs["Color"])
    links.new(emission_node.outputs["Emission"], output_node.inputs["Surface"])

    return mat

def assign_emissive_color_grid_material(obj):
    """
    Assign the emissive Color Grid material to the given object.
    """
    mat = create_emissive_color_grid_material()
    if not obj.data.materials:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat
        
def assign_emissive_color_grid_material_floor(obj):
    """
    Assign the emissive Color Grid material to the given object.
    """
    mat = create_emissive_color_grid_material_floor()
    if not obj.data.materials:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat

##################################################################
#               POINT SAMPLING FOR MESH OBJECTS                  #
##################################################################

def get_material_color(obj, face):
    """Return a default white color for now or attempt Principled BaseColor if present."""
    return [1.0, 1.0, 1.0]

def sample_points_from_object(obj, n_points):
    """
    Randomly sample points on the surface of obj using BMesh.
    Returns (pointsXYZ, colorsRGB).
    """
    bpy.ops.object.mode_set(mode='OBJECT')
    
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.transform(obj.matrix_world)
    
    bm.faces.ensure_lookup_table()
    num_faces = len(bm.faces)
    
    pts = []
    cols = []
    for _ in range(n_points):
        face = bm.faces[int(random() * num_faces)]
        if len(face.verts) < 3:
            continue
        if len(face.verts) == 3:
            rand_point = random_point_in_triangle(*face.verts)
        else:
            rand_point = random_point_in_polygon(face.verts)
        color = get_material_color(obj, face)
        pts.append(rand_point.to_tuple())
        cols.append(color)
    
    bm.free()
    return np.array(pts), np.array(cols)

##################################################################
#                       CAMERA SETUP                             #
##################################################################

def fibonacci_hemisphere(samples, sphere_radius):
    """Generate approximately uniform points on the upper hemisphere of radius."""
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # Golden angle
    for i in range(samples):
        z = i / float(samples - 1)  # 0 -> 1
        r = math.sqrt(1 - z*z)      # radius in xy-plane
        theta = phi * i
        x = r * math.cos(theta) * sphere_radius
        y = r * math.sin(theta) * sphere_radius
        z = z * sphere_radius
        points.append((x, y, z))
    return points

def look_at_matrix(camera_location, target=(0,0,0)):
    """Return a matrix that orients a camera at camera_location to look at 'target'."""
    loc = Vector(camera_location)
    tar = Vector(target)
    direction = tar - loc
    rot_quat = direction.to_track_quat('-Z', 'Y')
    m = rot_quat.to_matrix().to_4x4()
    m.translation = loc
    return m

def render_and_save(frame_idx, camera_idx, camera_obj, output_folder, prefix="cam", file_format="PNG"):
    """Render from camera_obj at current frame, saving to output_folder/camNNN/camNNN_renderFFFF.png."""
    cam_subfolder = os.path.join(output_folder, f"{prefix}{camera_idx:03d}")
    ensure_dir(cam_subfolder)

    filename = os.path.join(cam_subfolder, f"{prefix}{camera_idx:03d}_render{frame_idx:04d}.{file_format.lower()}")
    bpy.context.scene.camera = camera_obj
    bpy.context.scene.render.filepath = filename
    bpy.ops.render.render(write_still=True)

##################################################################
#                      MAIN SCRIPT LOGIC                         #
##################################################################

def main():
    # --------------------------------------------------------------------
    # 0. Global Blender Setup: Clear scene, set gravity, etc.
    # --------------------------------------------------------------------
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    scene = bpy.context.scene
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.scale_length = 1.0
    
    # Gravity
    scene.use_gravity = True
    scene.gravity = (0.0, 0.0, gravity)
    
    # Render settings
    setup_render_settings(res_x=600, res_y=600, samples=16, engine='BLENDER_EEVEE')
    
    # Rigid Body world settings for higher accuracy
    if not scene.rigidbody_world:
        bpy.ops.rigidbody.world_add()
    scene.rigidbody_world.enabled = True
    # Increase simulation accuracy:
    scene.rigidbody_world.substeps_per_frame = 20
    scene.rigidbody_world.solver_iterations = 100
    # Also enable split impulse for better bounce (less "sticking" at collisions)
    scene.rigidbody_world.use_split_impulse = True
    
    # --------------------------------------------------------------------
    # 1. Create a static thin floor
    # --------------------------------------------------------------------
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0,0,floor_location_z))
    floor_obj = bpy.context.active_object
    floor_obj.name = "Floor"
    floor_obj.scale = (floor_size, floor_size, floor_thickness*0.5)
    bpy.context.view_layer.update()
    
    # Apply scale so the collision shape matches
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    
    # Assign an emissive color grid material to floor
    assign_emissive_color_grid_material_floor(floor_obj)
    
    # Make floor a static rigid body
    bpy.ops.rigidbody.object_add()
    floor_obj.rigid_body.type = 'PASSIVE'
    floor_obj.rigid_body.friction = bounce_friction
    floor_obj.rigid_body.restitution = bounce_restitution
    floor_obj.rigid_body.collision_shape = 'MESH'  # 'MESH' sometimes can be finicky
    # Set collision margin for consistency
    floor_obj.rigid_body.collision_margin = collision_margin
    
    # --------------------------------------------------------------------
    # 2. Generate camera positions (on an upper hemisphere).
    #    We'll keep these camera objects for all sequences.
    # --------------------------------------------------------------------
    camera_positions = fibonacci_hemisphere(num_static_cams, camera_radius)
    camera_objects = []
    for i, cam_loc in enumerate(camera_positions):
        bpy.ops.object.camera_add()
        cam_obj = bpy.context.active_object
        cam_obj.name = f"Cam_{i:03d}"
        
        # Orient to look at (0,0,0)
        mat = look_at_matrix(cam_loc, (0,0,0))
        cam_obj.matrix_world = mat
        
        # Force 1:1 sensor
        cam_obj.data.sensor_width  = 36.0
        cam_obj.data.sensor_height = 36.0
        
        camera_objects.append(cam_obj)
    
    # --------------------------------------------------------------------
    # 3. Loop over multiple sequences
    # --------------------------------------------------------------------
    for seq_idx in range(1, num_sequences+1):
        print(f"=== Starting Sequence {seq_idx} ===")
        
        # Create a new collection for the sequence spheres
        collection_name = f"Sequence_{seq_idx}_Spheres"
        collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(collection)
        
        # 3a. Create and place spheres (random location, random radius)
        spheres = []
        attempts = 0
        while len(spheres) < num_spheres:
            attempts += 1
            if attempts > 2000:
                raise RuntimeError("Could not place spheres without intersection after many attempts.")

            radius = uniform(sphere_min_radius, sphere_max_radius)
            x = uniform(-floor_size*0.4, floor_size*0.4)
            y = uniform(-floor_size*0.4, floor_size*0.4)
            z = uniform(0.5+radius, max_initial_z)

            collide = False
            for s in spheres:
                dist = math.dist((x,y,z), (s.location.x, s.location.y, s.location.z))
                if dist < (s["radius"] + radius):
                    collide = True
                    break
            
            if not collide:
                # Create sphere
                bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=(x,y,z))
                sphere_obj = bpy.context.active_object
                sphere_obj.name = f"Sphere_{len(spheres):02d}"
                sphere_obj["radius"] = radius

                # Apply scale for correct collisions
                bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

                # Add to the new collection explicitly
                for c in sphere_obj.users_collection:
                    c.objects.unlink(sphere_obj)
                collection.objects.link(sphere_obj)

                # Assign an emissive color grid material
                assign_emissive_color_grid_material(sphere_obj)

                # Rigid body
                bpy.ops.rigidbody.object_add()
                sphere_obj.rigid_body.type = 'ACTIVE'
                sphere_obj.rigid_body.mass = 1.0
                sphere_obj.rigid_body.friction = bounce_friction
                sphere_obj.rigid_body.restitution = bounce_restitution
                sphere_obj.rigid_body.collision_margin = collision_margin
                sphere_obj.rigid_body.collision_shape = 'SPHERE'

                # **Disable damping** so they don't "stick" on collision
                sphere_obj.rigid_body.linear_damping = 0.0
                sphere_obj.rigid_body.angular_damping = 0.0
                
                # Also disable deactivation so they keep simulating
                sphere_obj.rigid_body.use_deactivation = False
                sphere_obj.rigid_body.deactivate_linear_velocity = 0.00001
                sphere_obj.rigid_body.deactivate_angular_velocity = 0.00001
                
                spheres.append(sphere_obj)
        
        # 3b. Bake the rigid body physics
        scene.frame_start = 0
        scene.frame_end   = num_frames
        bpy.context.view_layer.update()
        
        # Clear & bake
        bpy.ops.ptcache.free_bake_all()
        bpy.ops.ptcache.bake_all(bake=True)
        
        # 3c. Create output folder for this sequence
        seq_folder = os.path.join(base_output_folder, f"sequence{seq_idx}")
        ensure_dir(seq_folder)
        
        # 3d. Render frames
        cameras_data = []
        for frame_idx in range(num_frames):
            scene.frame_set(frame_idx)
            # Decide how many cameras to use:
            if frame_idx == 0:
                used_cams = camera_objects
            else:
                used_cams = camera_objects[:num_dynamic_cams]
            
            for cam_idx, cam_obj in enumerate(used_cams):
                c2w = cam_obj.matrix_world
                w2c = c2w.inverted()
                
                c2w_list = make_serializable(c2w)
                w2c_list = make_serializable(w2c)

                lens = cam_obj.data.lens
                width_px  = scene.render.resolution_x
                height_px = scene.render.resolution_y
                fx = lens * width_px  / cam_obj.data.sensor_width
                fy = lens * height_px / cam_obj.data.sensor_height

                cameras_data.append({
                    'sequence': seq_idx,
                    'frame': frame_idx,
                    'camera_id': cam_idx,
                    'w2c': w2c_list,
                    'c2w': c2w_list,
                    'img_name': f"cam{cam_idx:03d}_render{frame_idx:04d}",
                    'width': width_px,
                    'height': height_px,
                    'fx': fx,
                    'fy': fy
                })
                
                render_and_save(frame_idx=frame_idx,
                                camera_idx=cam_idx,
                                camera_obj=cam_obj,
                                output_folder=os.path.join(seq_folder, "ims"),
                                prefix="cam",
                                file_format="PNG")
        
        # Save cameras_gt.json
        cam_json_path = os.path.join(seq_folder, "cameras_gt.json")
        with open(cam_json_path, 'w') as jf:
            json.dump(cameras_data, jf, indent=2)
        
        # 3e. Sample geometry from the floor + spheres => points3d.ply
        all_pts = []
        all_cols= []

        # Floor
        floor_pts, floor_cols = sample_points_from_object(floor_obj, n_points_per_mesh)
        all_pts.append(floor_pts)
        all_cols.append(floor_cols)

        # Each sphere
        for sp in spheres:
            sp_pts, sp_cols = sample_points_from_object(sp, n_points_per_mesh)
            all_pts.append(sp_pts)
            all_cols.append(sp_cols)

        all_pts  = np.concatenate(all_pts, axis=0)
        all_cols = np.concatenate(all_cols, axis=0)

        # PLY with [x, y, z, nx, ny, nz, r, g, b]
        verts = np.empty(len(all_pts),
                         dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        verts['x'] = all_pts[:, 0]
        verts['y'] = all_pts[:, 1]
        verts['z'] = all_pts[:, 2]
        verts['nx'] = 0
        verts['ny'] = 0
        verts['nz'] = 0
        
        # Convert colors to [0..255]
        r = np.clip(all_cols[:,0]*255, 0, 255).astype(np.uint8)
        g = np.clip(all_cols[:,1]*255, 0, 255).astype(np.uint8)
        b = np.clip(all_cols[:,2]*255, 0, 255).astype(np.uint8)
        verts['red']   = r
        verts['green'] = g
        verts['blue']  = b
        
        from plyfile import PlyData, PlyElement
        el = PlyElement.describe(verts, 'vertex')
        ply_path = os.path.join(seq_folder, "points3d.ply")
        PlyData([el]).write(ply_path)
        
        # 3f. Remove spheres before starting next sequence
        for sp in spheres:
            bpy.data.objects.remove(sp, do_unlink=True)
        bpy.data.collections.remove(collection)
        
        bpy.ops.ptcache.free_bake_all()

    print("All sequences generated successfully!")

# ----------------------------------------------------------------
# Run the main logic
# ----------------------------------------------------------------
if __name__ == "__main__":
    main()
