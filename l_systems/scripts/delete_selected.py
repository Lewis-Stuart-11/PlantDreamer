import bpy

def delete_selected_objects():
    bpy.data.batch_remove(bpy.context.selected_objects)
    
#delete_selected_objects()
