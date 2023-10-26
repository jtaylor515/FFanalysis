import subprocess
from vars import *

# List of packages to install
if gpu == True:
    packages_to_install = ['nbformat', 'pandas', 'requests', 'bs4', 'nbformat', 'seaborn', 'scikit-learn', 'cudf', 'cuml', 'cupy', 'numpy']
else:
    packages_to_install = ['nbformat', 'pandas', 'requests', 'bs4', 'nbformat', 'seaborn', 'scikit-learn', 'numpy']

for package in packages_to_install:
    try:
        # Use subprocess to run pip install command
        subprocess.check_call(['pip', 'install', package, '-q'])
        print(f"Successfully installed {package}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}: {e}")