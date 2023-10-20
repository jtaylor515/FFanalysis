import subprocess
import os

# Run 'updateScript.py'
subprocess.run(["python", "updateScript.py"])

# Get the current working directory
current_directory = os.getcwd()

# Directory where the Python scripts are located
script_directory = "python_scripts"

# Combine the current directory and the script directory
full_directory = os.path.join(current_directory, script_directory)

# List all Python script files in the specified directory
python_script_files = [f for f in os.listdir(full_directory) if f.endswith(".py")]
print(python_script_files)

for script_file in python_script_files:
    script_path = os.path.join(full_directory, script_file)
        
    # Run each Python script using subprocess.run with the full path
    subprocess.run(["python", script_path])

print("Dataset Update and Prediction Complete")
