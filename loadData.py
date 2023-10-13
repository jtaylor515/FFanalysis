### CODE CELL ###
import pandas as pd
import requests
from bs4 import BeautifulSoup
### END OF CODE CELL ###

### CODE CELL ###
# List of URLs to scrape
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

# Concatenate values from rows 1 and 2 into a new row 3
merged_df.loc[2] = merged_df.loc[0] + ' ' + merged_df.loc[1]

# Remove rows 1 and 2
merged_df = merged_df.drop([0, 1])

# Reset the index
merged_df.reset_index(drop=True, inplace=True)

# Display the merged DataFrame
print(merged_df)

# If you want to save the data to a CSV file, you can do it like this:
merged_df.to_csv('fantasy_stats.csv', index=False)
### END OF CODE CELL ###

