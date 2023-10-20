### CODE CELL ###
import importlib
import pandas as pd
import requests
from bs4 import BeautifulSoup
import nbformat
from io import StringIO
import os
### END OF CODE CELL ###

### CODE CELL ###
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

# Get the current working directory
current_directory = os.getcwd()

# Get the parent directory
parent_directory = os.path.dirname(current_directory)

# Change the working directory to the parent directory
os.chdir(parent_directory)

merged_df.to_csv('datasets/overall_scoring.csv', index=False)
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
snap_count_merged_df.to_csv('datasets/snap_counts.csv', index=False)

snap_count_merged_df.head(10)
### END OF CODE CELL ###

### CODE CELL ###
# Get user input for the number of weeks to scrape
num_weeks = int(input("Enter the current week: "))

# Define the base URL and the page options
base_url = "https://www.fantasypros.com/nfl/stats/"

# List of page options
pages = ['qb.php', 'wr.php', 'rb.php', 'te.php']

# Initialize the list to store the generated URLs
urls = []

# Generate URLs based on user input for the 2020, 2021, 2022, and 2023 seasons
for page in pages:
    for year in range(2018, 2024):  # Loop through years 2020, 2021, 2022, and 2023
        max_week = 18 if year > 2020 else 17  # Weeks 1-17 for 2020, Weeks 1-18 for 2021 and 2022
        for week in range(1, max_week + 1):
            if year == 2023 and week > num_weeks: # Weeks 1-INPUT for 2023
                break  # Stop generating URLs for 2022 if the desired week is reached
            url = f"{base_url}{page}?year={year}&range=week&week={week}"
            urls.append(url)

# Print the list of generated URLs
for url in urls:
    print(url)

### END OF CODE CELL ###

### CODE CELL ###
# Initialize an empty DataFrame to store the data
final_dataset = pd.DataFrame()

# Iterate through the URLs
for url in urls:
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Assuming the data is in a table, you may need to adjust the code based on the actual structure
        table = soup.find('table')

        # Use io.StringIO to wrap the HTML content
        table_string = str(table)
        table_io = StringIO(table_string)
        
        # Read the table into a DataFrame
        df = pd.read_html(table_io)[0]

        # Add a "LOC" column to the DataFrame
        loc = url.split('/')[-1][:2]
        df[("LOC", "POS")] = loc
        
        # Extract week value from the URL
        week_value = int(url.split('week=')[1])
        
        # Add a new 'Week' column with the week value
        df['WEEK'] = week_value
        
        # Concatenate the DataFrame to the final dataset
        final_dataset = pd.concat([final_dataset, df], ignore_index=True)
    else:
        print(f"Failed to fetch data from URL: {url}")

# Now, final_dataset contains the combined data with a 'Week' column
# final_dataset.head(10)

# Combine values in column names (headers) and row 0
final_dataset.columns = final_dataset.columns.map(' '.join)

# Reset the index
final_dataset.reset_index(drop=True, inplace=True)

# Rename columns as specified
final_dataset = final_dataset.rename(columns={"Unnamed: 0_level_0 Rank": "POS RANK", "Unnamed: 1_level_0 Player": "PLAYER", "LOC POS": "POS", "WEEK ": "WEEK"})

final_dataset.to_csv('datasets/weekly_scoring.csv', index=False)

# final_dataset.head(10)


## PRINT DATAFRAME DATA
# Assuming you have a DataFrame named 'df'
# Get the number of rows
num_rows = final_dataset.shape[0]

# Get the number of columns
num_columns = final_dataset.shape[1]

# Calculate the product of rows and columns
rows_times_columns = num_rows * num_columns

# Print the results
print(f"Number of Rows: {num_rows}")
print(f"Number of Columns: {num_columns}")
print(f"Rows times Columns: {rows_times_columns}")

import os

file_path = "datasets/weekly_scoring.csv"  # Replace with the path to your CSV file

# Check if the file exists
if os.path.exists(file_path):
    # Get the size of the file in bytes
    file_size_bytes = os.path.getsize(file_path)

    # Convert bytes to megabytes
    file_size_mb = file_size_bytes / (1024 * 1024)

    print(f"File Size: {file_size_mb:.2f} MB")
else:
    print("File not found.")
### END OF CODE CELL ###

### CODE CELL ###
import subprocess

# List of file paths to push
file_paths = ["datasets/"]

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

