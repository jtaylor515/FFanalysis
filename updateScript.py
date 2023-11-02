import subprocess
import nbformat
import os

# List of input Jupyter Notebook files
input_nb_files = [
    "DataPreprocess.ipynb",
    # "DataPostprocess.ipynb",
    "RandomForest.ipynb",
    "LinearRegression.ipynb"
    # add more files as needed
]

# Output directory for Python script files
output_directory = os.getcwd()  # Use the current working directory

def extract_code_cells(input_file, output_directory):
    with open(input_file, "r") as notebook_file:
        notebook = nbformat.read(notebook_file, as_version=4)

    output_file = os.path.splitext(os.path.basename(input_file))[0] + ".py"

    with open(output_file, "w") as python_file:
        for cell in notebook.cells:
            if cell.cell_type == "code":
                python_file.write(f"### CODE CELL ###\n")
                for line in cell.source.splitlines():
                    python_file.write(f"{line}\n")
                python_file.write(f"### END OF CODE CELL ###\n\n")

if __name__ == "__main__":
    for input_nb_file in input_nb_files:
        extract_code_cells(input_nb_file, output_directory)

    # Remove the "python_scripts" directory if it exists
    if os.path.exists("python_scripts"):
        shutil.rmtree("python_scripts")

    print("Files are updated. RUNME.py can continue running")