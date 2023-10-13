import nbformat

# Specify the input Jupyter Notebook file
input_nb_file = "DataPreprocess.ipynb"

# Specify the output Python script file
output_py_file = "loadData.py"

def extract_code_cells(input_file, output_file):
    with open(input_file, "r") as notebook_file:
        notebook = nbformat.read(notebook_file, as_version=4)
    
    with open(output_file, "w") as python_file:
        for cell in notebook.cells:
            if cell.cell_type == "code":
                python_file.write(f"### CODE CELL ###\n")
                for line in cell.source.splitlines():
                    python_file.write(f"{line}\n")
                python_file.write(f"### END OF CODE CELL ###\n\n")

if __name__ == "__main__":
    extract_code_cells(input_nb_file, output_py_file)
