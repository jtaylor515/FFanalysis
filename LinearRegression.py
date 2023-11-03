### CODE CELL ###
from vars import *
### END OF CODE CELL ###

### CODE CELL ###
if gpu == False: 
    # Get user input for the week number to predict
    num_week = int(input("Enter the week to predict: "))

    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import LabelEncoder


    for pos in positions:
        # Load your dataset
        data = pd.read_csv("datasets/weekly_scoring.csv")

        # Preprocessing
        data = data[data['POS'] == pos]
        weights = data['WEIGHT']
        data = data.drop(columns=['POS RANK', 'POS', 'MISC G', 'MISC ROST', 'MISC FPTS/G', 'RECEIVING REC', 'RECEIVING TGT', 'RECEIVING YDS', 'RECEIVING Y/R',
        'RECEIVING LG', 'RECEIVING 20+', 'RECEIVING TD', 'RUSHING Y/A', 'RUSHING LG',
        'RUSHING 20+', 'DATE', 'YEAR', 'WEIGHT'])
        data = pd.get_dummies(data, columns=['PLAYER'], drop_first=True)

        # Identify columns with missing values before imputation
        columns_with_missing = data.columns[data.isnull().any()].tolist()

        # Impute missing values with the mean of each column
        imputer = SimpleImputer(strategy='mean')
        data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

        # Define the list of variables to predict :)
        var_list = ['PASSING CMP', 'PASSING ATT', 'PASSING PCT', 'PASSING YDS', 'PASSING Y/A', 'PASSING TD', 'PASSING INT',
                    'PASSING SACKS', 'RUSHING ATT', 'RUSHING YDS', 'RUSHING TD', 'MISC FL', 'MISC FPTS', 'WEEK']

        # Separate the dataset into features (X) and the target variable (y)
        X = data.drop(var_list, axis=1)
        y = data['MISC FPTS']

        # Save a copy of the dataset
        df = data

        # Hyperparameter tuning using GridSearchCV for Linear Regression
        param_grid = {
            'fit_intercept': [True, False]
        }

        grid_search = GridSearchCV(LinearRegression(), param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X, y)

        # Fit the model with the best hyperparameters
        best_lr_model = grid_search.best_estimator_
        best_lr_model.fit(X, y)

        # Get a list of unique player names after one-hot encoding
        unique_players = X.columns

        # Create a list of dictionaries to store the results
        results_list = []

        for player in unique_players:
            # Create a DataFrame with all zeros
            week6_data = pd.DataFrame(0, index=range(1), columns=X.columns)
            # Set the corresponding player's column to 1 for prediction
            week6_data[player] = 1
            # Make a prediction for the player
            misc_fpts_prediction = best_lr_model.predict(week6_data)
            results_list.append({'Player': player, 'MISC FPTS': misc_fpts_prediction[0]})

        # Convert the list of dictionaries into a DataFrame
        results_df = pd.DataFrame(results_list)

        results_df = results_df.sort_values(by='MISC FPTS', ascending=False)

        # Save the results to a CSV file
        file_name = f"predictions/LRweek{num_week}{pos}unweighted.csv"
        results_df.to_csv(file_name, index=False)
### END OF CODE CELL ###

### CODE CELL ###
if gpu == True:
    # Get user input for the week number to predict
    num_week = int(input("Enter the week to predict: "))

    import cudf
    from cuml.linear_model import LinearRegression
    from cuml.model_selection import GridSearchCV
    from cuml.impute import SimpleImputer
    from cuml.preprocessing import LabelEncoder



    for pos in positions:
        # Load your dataset using cuDF
        data = cudf.read_csv("datasets/weekly_scoring.csv")
        
        # Preprocessing
        data = data[data['POS'] == pos]
        data = data.drop(columns=['POS RANK', 'POS', 'MISC G', 'MISC ROST', 'MISC FPTS/G', 'RECEIVING REC', 'RECEIVING TGT', 'RECEIVING YDS', 'RECEIVING Y/R',
        'RECEIVING LG', 'RECEIVING 20+', 'RECEIVING TD', 'RUSHING Y/A', 'RUSHING LG',
        'RUSHING 20+'])
        data = cudf.get_dummies(data, columns=['PLAYER'], drop_first=True)

        # Identify columns with missing values before imputation
        columns_with_missing = data.columns[data.isnull().any()].tolist()

        # Impute missing values with the mean of each column
        imputer = SimpleImputer(strategy='mean')
        data = cudf.DataFrame(imputer.fit_transform(data), columns=data.columns)

        # Define the list of variables to predict
        var_list = ['PASSING CMP', 'PASSING ATT', 'PASSING PCT', 'PASSING YDS', 'PASSING Y/A', 'PASSING TD', 'PASSING INT',
                    'PASSING SACKS', 'RUSHING ATT', 'RUSHING YDS', 'RUSHING TD', 'MISC FL', 'MISC FPTS', 'WEEK']

        # Separate the dataset into features (X) and the target variable (y)
        X = data.drop(var_list, axis=1)
        y = data['MISC FPTS']

        # Save a copy of the dataset
        df = data

        # Hyperparameter tuning using GridSearchCV for Linear Regression
        param_grid = {
            'fit_intercept': [True, False]
        }

        grid_search = GridSearchCV(LinearRegression(), param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X, y)

        # Fit the model with the best hyperparameters
        best_lr_model = grid_search.best_estimator_
        best_lr_model.fit(X, y)

        # Get a list of unique player names after one-hot encoding
        unique_players = X.columns

        # Create a list of dictionaries to store the results
        results_list = []

        for player in unique_players:
            # Create a DataFrame with all zeros
            week6_data = cudf.DataFrame(0, index=range(1), columns=X.columns)
            # Set the corresponding player's column to 1 for prediction
            week6_data[player] = 1
            # Make a prediction for the player
            misc_fpts_prediction = best_lr_model.predict(week6_data)
            results_list.append({'Player': player, 'MISC FPTS': misc_fpts_prediction[0]})

        # Convert the list of dictionaries into a DataFrame
        results_df = cudf.DataFrame(results_list)

        results_df.head(10)

        # Save the results to a CSV file
        file_name = f"predictions/LRweek{num_week}{pos}.csv"
        results_df.to_pandas().to_csv(file_name, index=False)
### END OF CODE CELL ###

