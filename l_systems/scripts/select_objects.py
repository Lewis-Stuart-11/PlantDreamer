import bpy

def select_objects_except_in_collection(exclude_collection):
  """Selects all objects except those in a collection"""
  #deselect_all_objects()
  exclude_collection = exclude_collection.lower().strip()
  bpy.ops.object.select_all(action='SELECT')
  for obj in bpy.data.objects:
    if exclude_collection in [c.name.lower().strip() for c in obj.users_collection]:
      obj.select_set(False)
    else:
      try:
        obj.select_set(True)
      except Exception:
        pass
      
def deselect_all_objects():
  """Selects all objects except those with names that exactly match strings in the exclude_list."""
  bpy.ops.object.select_all(action='SELECT')
  for obj in bpy.data.objects:
      obj.select_set(False)  


# Example Usage:
#exclude_objects_for_exports = ["kleaf"] #list of objects to exclude
#select_objects_except_exact_matches(exclude_objects_for_exports)