"""
Blender batch render script.
Run inside Blender: Scripting panel -> paste and run.
Or headless: blender --background model.blend --python blender_render.py
"""
import bpy
import math
import json
import os

OUTPUT_DIR = "/path/to/data/train/laukry"   # CHANGE THIS
N_VIEWS = 100
RESOLUTION = 512
ELEVATION_DEG = 20.0

os.makedirs(f"{OUTPUT_DIR}/multiview", exist_ok=True)

# Scene setup
scene = bpy.context.scene
scene.render.resolution_x = RESOLUTION
scene.render.resolution_y = RESOLUTION
scene.render.image_settings.file_format = 'PNG'
scene.render.film_transparent = True   # RGBA, transparent background

# Place camera on sphere
cam = bpy.data.objects.get("Camera") or bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
bpy.context.scene.collection.objects.link(cam)
scene.camera = cam

radius = 2.5
cameras = []

for i in range(N_VIEWS):
    azimuth = (360.0 / N_VIEWS) * i
    elevation = ELEVATION_DEG

    az_rad = math.radians(azimuth)
    el_rad = math.radians(elevation)

    x = radius * math.cos(el_rad) * math.cos(az_rad)
    y = radius * math.cos(el_rad) * math.sin(az_rad)
    z = radius * math.sin(el_rad)

    cam.location = (x, y, z)

    # Point camera at origin
    direction = cam.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()

    # Render
    filepath = f"{OUTPUT_DIR}/multiview/{i:03d}.png"
    scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)

    cameras.append({
        "id": i,
        "azimuth": azimuth,
        "elevation": elevation,
        "image": f"multiview/{i:03d}.png",
        "camera_matrix": [list(row) for row in cam.matrix_world]
    })
    print(f"Rendered {i+1}/{N_VIEWS}: azimuth={azimuth:.1f}")

# Save cameras.json
with open(f"{OUTPUT_DIR}/cameras.json", "w") as f:
    json.dump({"views": cameras}, f, indent=2)

print(f"Done. Saved {N_VIEWS} views to {OUTPUT_DIR}")
