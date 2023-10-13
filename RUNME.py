import subprocess
import os

# Directory where the Python scripts are located
script_directory = "python_scripts"

if __name__ == "__main":
    # Run 'updateScript.py'
    subprocess.run(["python", "updateScript.py"])

    # List all Python script files in the specified directory
    python_script_files = [f for f in os.listdir(script_directory) if f.endswith(".py")]

    for script_file in python_script_files:
        script_path = os.path.join(script_directory, script_file)
        
        # Run each Python script using subprocess.run
        subprocess.run(["python", script_path])
