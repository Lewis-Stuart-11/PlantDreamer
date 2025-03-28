import bpy

def export_selected_as_ply(filepath):
    selected_objects = bpy.context.selected_objects

    if not selected_objects:
        print("No objects selected")
        return

    try:
        bpy.ops.wm.ply_export(
            filepath=filepath,
            forward_axis='Y',
            up_axis='Z',
            global_scale=1.0,
            apply_modifiers=True,
            export_selected_objects=True,  # Ensures only selected objects are exported
            export_uv=True,
            export_normals=True,  # Enable normal export (was False in documentation)
            export_colors='SRGB',  # Exports vertex colors in SRGB space
            export_attributes=True,  # Include attributes in the PLY file
            export_triangulated_mesh=False,
            ascii_format=False,  # Set to True if you need an ASCII PLY file instead of binary
            filter_glob='*.ply'
        )
        print(f"Exported selected objects to: {filepath}")

    except Exception as e:
        print(f"Error exporting PLY: {e}")
