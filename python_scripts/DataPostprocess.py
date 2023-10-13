### CODE CELL ###
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
### END OF CODE CELL ###

### CODE CELL ###
# Example GitHub repository URL with raw CSV file
github_csv_url = 'https://raw.githubusercontent.com/jtaylor515/FFanalysis/main/overall_scoring.csv'
github_csv_url2 = 'https://raw.githubusercontent.com/jtaylor515/FFanalysis/main/snap_counts.csv'


# Read the CSV file into a Pandas DataFrame
overall_scoring = pd.read_csv(github_csv_url)
snap_counts = pd.read_csv(github_csv_url2)

overall_scoring.head(10)
### END OF CODE CELL ###

### CODE CELL ###
# Sort the DataFrame by the "FPTS" column in descending order
sorted_df = overall_scoring.sort_values(by=[("MISC FPTS")], ascending=False)

# Display the sorted DataFrame
sorted_df.head(10)
### END OF CODE CELL ###

### CODE CELL ###
# Filter out the "POS RANK" and "PLAYER" columns
filtered_overall_scoring = overall_scoring.drop(columns=["POS RANK", "PLAYER", "MISC ROST", "MISC FPTS/G"])
### END OF CODE CELL ###

### CODE CELL ###
# Initialize an empty list to store results for each position
results_per_position = []

# Loop through unique values in the "POS" column
for pos_value in filtered_overall_scoring["POS"].unique():
    # Filter the data for the current position
    filtered_data = filtered_overall_scoring[filtered_overall_scoring["POS"] == pos_value]

    filtered_data = filtered_data.drop(columns=["POS"])
    
    # Calculate the correlation matrix for the filtered data
    correlation_matrix = filtered_data.corr()
    
    # Get the correlation of "MISC FPTS" with all columns, excluding itself
    misc_fpts_correlation = correlation_matrix.loc["MISC FPTS"].drop("MISC FPTS")
    
    # Sort the correlations in descending order
    sorted_correlation = misc_fpts_correlation.sort_values(ascending=False)
    
    # Get the columns with the highest correlation
    highest_correlation_columns = sorted_correlation.index
    
    # Store the result for this position in the list
    results_per_position.append((pos_value, highest_correlation_columns))

    # Create a heatmap of the correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title("Correlation Matrix Heatmap for " + pos_value)
    plt.show()


# Print the top 10 results for each position
for pos_value, top_correlation_columns in results_per_position:
    print(f"Top 10 stats with the highest correlation to total FPTS for position {pos_value} in descending order:")
    print(top_correlation_columns[1:10])
    print("\n")
### END OF CODE CELL ###

### CODE CELL ###
# Filter out the "POS RANK" and "PLAYER" columns
filtered_overall_scoring = overall_scoring.drop(columns=["POS RANK", "PLAYER", "MISC ROST", "MISC FPTS"])
### END OF CODE CELL ###

### CODE CELL ###
# Initialize an empty list to store results for each position
results_per_position = []

# Loop through unique values in the "POS" column
for pos_value in filtered_overall_scoring["POS"].unique():
    # Filter the data for the current position
    filtered_data = filtered_overall_scoring[filtered_overall_scoring["POS"] == pos_value]

    filtered_data = filtered_data.drop(columns=["POS"])
    
    # Calculate the correlation matrix for the filtered data
    correlation_matrix = filtered_data.corr()
    
    # Get the correlation of "MISC FPTS" with all columns, excluding itself
    misc_fpts_correlation = correlation_matrix.loc["MISC FPTS/G"].drop("MISC FPTS/G")
    
    # Sort the correlations in descending order
    sorted_correlation = misc_fpts_correlation.sort_values(ascending=False)
    
    # Get the columns with the highest correlation
    highest_correlation_columns = sorted_correlation.index
    
    # Store the result for this position in the list
    results_per_position.append((pos_value, highest_correlation_columns))

    # Create a heatmap of the correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title("Correlation Matrix Heatmap for " + pos_value)
    plt.show()


# Print the top 10 results for each position
for pos_value, top_correlation_columns in results_per_position:
    print(f"Top 10 stats with the highest correlation to total FPTS for position {pos_value} in descending order:")
    print(top_correlation_columns[1:10])
    print("\n")
### END OF CODE CELL ###

