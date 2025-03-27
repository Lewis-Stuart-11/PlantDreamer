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

    # Check if Blender is installed
    if not shutil.which("blender"):
        raise EnvironmentError("Blender is not installed or not found in system PATH.")

    # Run the script in Blender
    try:
        subprocess.run(
            ["blender", "--background", "--python", str(l_system_script)], 
            check=True  # Ensures an error is raised if the process fails
        )
    except subprocess.CalledProcessError as e:
        print(f"Error executing Blender script: {e}")
        raise

if __name__ == "__main__":
    parser = configargparse.ArgumentParser()

    parser.add_argument("--plant_species", type=str, default="", required=True, help="The name of the plant species")

    args = parser.parse_args()

    get_l_system(args.plant_species)
    

