### CODE CELL ###
import pandas as pd
import requests
from bs4 import BeautifulSoup
import subprocess
import nbformat
### END OF CODE CELL ###

### CODE CELL ###
import pandas as pd
import requests
from bs4 import BeautifulSoup

# List of URLs to scrape for fantasy stats
urls = [
    'https://www.fantasypros.com/nfl/stats/qb.php?scoring=HALF&roster=y',
    'https://www.fantasypros.com/nfl/stats/rb.php?scoring=HALF&roster=y',
    'https://www.fantasypros.com/nfl/stats/wr.php?scoring=HALF&roster=y',
    'https://www.fantasypros.com/nfl/stats/te.php?scoring=HALF&roster=y'
]

# Initialize an empty list to store DataFrames
data_frames = []

for url in urls:
    # Send an HTTP GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the table on the page
        table = soup.find('table')

        # Read the table into a Pandas DataFrame
        df = pd.read_html(str(table), header=[0, 1])[0]
        
        # Add a "LOC" column to the DataFrame
        loc = url.split('/')[-1][:2]
        df[("LOC", "POS")] = loc

        data_frames.append(df)
    else:
        print(f"Failed to retrieve data from {url}")

# Merge all DataFrames into one based on the first and second row headers
merged_df = pd.concat(data_frames, ignore_index=True)

# Combine values in column names (headers) and row 0
merged_df.columns = merged_df.columns.map(' '.join)

# Reset the index
merged_df.reset_index(drop=True, inplace=True)

# Rename columns as specified
merged_df = merged_df.rename(columns={"Unnamed: 0_level_0 Rank": "POS RANK", "Unnamed: 1_level_0 Player": "PLAYER", "LOC POS": "POS"})

merged_df.to_csv('overall_scoring.csv', index=False)
### END OF CODE CELL ###

### CODE CELL ###
import pandas as pd
import requests
from bs4 import BeautifulSoup

# List of URLs to scrape for snap counts
snap_count_urls = [
    'https://www.fantasypros.com/nfl/reports/snap-counts/rb.php?show=perc',
    'https://www.fantasypros.com/nfl/reports/snap-counts/wr.php?show=perc',
    'https://www.fantasypros.com/nfl/reports/snap-counts/te.php?show=perc'
]

# Initialize an empty list to store DataFrames for snap counts
snap_count_data_frames = []

for url in snap_count_urls:
    # Send an HTTP GET request to the URL for snap counts
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the table on the page
        table = soup.find('table')

        # Read the table into a Pandas DataFrame
        df = pd.read_html(str(table), header=[0])[0]
        
        # Add a "POS" column to the DataFrame for snap counts
        pos = url.split('/')[-1][:2]
        df[("POS")] = pos

        snap_count_data_frames.append(df)
    else:
        print(f"Failed to retrieve data from {url} (snap counts)")

# Concatenate (append) all DataFrames for snap counts
snap_count_merged_df = pd.concat(snap_count_data_frames, ignore_index=True)

# If you want to save the data to a CSV file, you can do it like this:
snap_count_merged_df.to_csv('snap_counts.csv', index=False)

snap_count_merged_df.head(10)
### END OF CODE CELL ###

### CODE CELL ###
import subprocess

# List of file paths to push
file_paths = ["overall_scoring.csv", "snap_counts.csv"]

# Specify the GitHub repository URL
repo_url = "https://github.com/jtaylor515/FFanalysis.git"

# Specify your commit message
commit_message = "Update files"

# Git commands to add, commit, and push each file in the list
for file_path in file_paths:
    try:
        subprocess.run(["git", "add", file_path])
        subprocess.run(["git", "commit", "-m", commit_message])
        subprocess.run(["git", "push", repo_url])
        print(f"File {file_path} successfully pushed to the repository.")
    except Exception as e:
        print(f"Error: {e}")
### END OF CODE CELL ###

