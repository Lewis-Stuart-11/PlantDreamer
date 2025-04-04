import configargparse
import subprocess
import shutil
from pathlib import Path

def get_l_system(plant_species):
    plant_type_str = plant_species.lower().strip()

    # Construct the L-System script path
    l_system_path = Path(__file__).parent.with_name("l_systems")
    l_system_script = l_system_path / f"{plant_type_str}.py"

    # Check if the script exists
    if not l_system_script.exists():
        raise FileNotFoundError(f"L-System script '{l_system_script}' has not been implemented.")

    # Construct blender file path
    blender_l_system_path = l_system_path / f"blender_models" / f"{plant_type_str}.blend"
    
    # Check if the blender plant exists
    if not l_system_script.exists():
        raise FileNotFoundError(f"L-System blender file '{blender_l_system_path}' has not been implemented.")

    # Check if Blender is installed
    if not shutil.which("blender"):
        raise EnvironmentError("Blender is not installed or not found in system PATH.")

    # Run the script in Blender
    try:
        subprocess.run(
            ["blender", "--background", str(blender_l_system_path), "--python", str(l_system_script)], 
            check=True  
        )
    except subprocess.CalledProcessError as e:
        print(f"Error executing Blender script: {e}")
        raise

    # Get generated L_System mesh and check it exists
    output_path = l_system_path / "output" / f"plant.ply"
    
    if not output_path.exists():
        raise FileNotFoundError(f"Could not find generated L-System mesh in directory {output_path}")

    return str(output_path)

if __name__ == "__main__":
    parser = configargparse.ArgumentParser()

    parser.add_argument("--plant_species", type=str, default="", required=True, help="The name of the plant species")

    args = parser.parse_args()

    get_l_system(args.plant_species)
    

