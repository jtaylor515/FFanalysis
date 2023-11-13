import subprocess
import os

## INSTALL ALL NECESSARY PACKAGES
# Run the command to install packages from requirements.txt
try:
    subprocess.check_call(['pip', 'install', '-r', 'requirements.txt', '-q'], shell = True)
    print("Successfully installed packages from requirements.txt")
except subprocess.CalledProcessError as e:
    print(f"Failed to install packages from requirements.txt: {e}")

# Run 'updateScript.py'
subprocess.run(["python", "updateScript.py"])

# Get the current working directory
current_directory = os.getcwd()

# List all Python script files in the specified directory
python_script_files = [f for f in os.listdir(current_directory) if f.endswith(".py")]
print(python_script_files)

# List of Python script files to run
python_script_files = [
    # "DataPreprocess.py"
    # "LinearRegression.py",
    # "RandomForest.py",
    # Add more script filenames as needed
]

current_directory = os.getcwd()

for script_file in python_script_files:
    script_path = os.path.join(current_directory, script_file)

    # Run each Python script using subprocess.run with the full path
    print("Running", script_file)
    subprocess.run(["python", script_path])

print("Dataset Update and Prediction Complete")
