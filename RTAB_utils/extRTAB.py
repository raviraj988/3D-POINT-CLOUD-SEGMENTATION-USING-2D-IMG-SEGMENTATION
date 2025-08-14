import subprocess
import os
import shutil



def exportRTAB(path_in, path_out):
    """exportRTAB function helps for extraction of 3d data from the RTAB database type.

    Args:
        path_in (string): Path for the Database file created using RTAB Map
        path_out (string): Path for saving the database contents that exports [calib, depth, rgb, cloud]

    Returns:
        bool: True for successful extraction, False for unsuccessful extraction
    """
    print("Exporting 3D RTAB Data")
    
    # Check if path_in exists
    if not os.path.exists(path_in):
        print(f"Error: Path '{path_in}' does not exist.")
        return False

    # Check if path_out exists
    if os.path.exists(path_out):
        print(f"Removing existing directory: '{path_out}'")
        shutil.rmtree(path_out)

    # Create path_out directory
    print(f"Creating directory: '{path_out}'")
    os.makedirs(path_out)

    # Construct command
    command = f'rtabmap-export --images --poses_format 11 --ba --poses_camera --images_id --output_dir "{path_out}" "{path_in}"'

    # Execute command
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Print command output
    print("Command Output:")
    print(result.stdout)

    # Check if the extraction was successful
    if result.returncode == 0:
        print("Export successful.")
        return True
    else:
        print("Export failed.")
        return False