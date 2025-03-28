import bpy
import os

def generate_mesh(l_system, exclude_objects_with_collection="l_system"):
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(output_dir, exist_ok=True)

    output_file =  os.path.join(output_dir, "plant.ply")
    if os.path.exists(output_file):
        os.remove(output_file)

    # Import external modules
    import scripts.select_objects as select_objects
    import scripts.export as export
    import scripts.delete_selected as delete_selected

    print("Starting Script")

    # Deselect all objects
    select_objects.deselect_all_objects()

    print("Generating Plant")
    # Generate the L-System
    l_system()

    select_objects.select_objects_except_in_collection(exclude_objects_with_collection)

    print("Exporting Plant")
    # Export PLY file to specific filepath
    export.export_selected_as_ply(output_file)

    """# Select objects to be deleted
    exclude_objects_for_deletion = ["kleaf", "Pot", "Soil"]
    select_objects.select_objects_except_exact_matches(exclude_objects_for_deletion)

    print("Deleting Plant")
    delete_selected.delete_selected_objects()

    print("Generated Plant")"""
