### CODE CELL ###
from vars import *
### END OF CODE CELL ###

### CODE CELL ###
if gpu == False:

    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.impute import SimpleImputer

    # Get user input for the week number to predict
    num_week = int(input("Enter the week to predict: "))

    for pos in positions:
        # Load your dataset
        data = pd.read_csv("datasets/weekly_scoring.csv")

        # Preprocessing
        data = data[data['POS'] == pos]
        weights = data['WEIGHT']
        data = data.drop(columns=['POS RANK', 'POS', 'MISC G', 'MISC ROST', 'MISC FPTS/G', 'RECEIVING REC', 'RECEIVING TGT', 'RECEIVING YDS', 'RECEIVING Y/R',
        'RECEIVING LG', 'RECEIVING 20+', 'RECEIVING TD', 'RUSHING Y/A', 'RUSHING LG', 'RUSHING 20+', 'DATE', 'YEAR', 'WEIGHT', 'WEEK'])
        data = pd.get_dummies(data, columns=['PLAYER'], drop_first=True)

        # Identify columns with missing values before imputation
        columns_with_missing = data.columns[data.isnull().any()].tolist()

        # Impute missing values with the mean of each column
        # imputer = SimpleImputer(strategy='mean')
        imputer = SimpleImputer(strategy='median')  # Use 'median' strategy

        data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

        # Define the list of variables to predict
        var_list = ['PASSING CMP', 'PASSING ATT', 'PASSING PCT', 'PASSING YDS', 'PASSING Y/A', 'PASSING TD', 'PASSING INT',
                    'PASSING SACKS', 'RUSHING ATT', 'RUSHING YDS', 'RUSHING TD', 'MISC FL', 'MISC FPTS']

        # Separate the dataset into features (X) and the target variable (y)
        X = data.drop(var_list, axis=1)
        y = data['MISC FPTS']

        # Save a copy of the dataset
        df = data

        # Hyperparameter tuning using GridSearchCV
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X, y)

        # Fit the model with the best hyperparameters
        best_rf_model = grid_search.best_estimator_
        # best_rf_model.fit(X, y)

        # Fit the model with weighted on date
        best_rf_model.fit(X, y, sample_weight=weights)

        # Get a list of unique player names after one-hot encoding
        unique_players = X.columns

        # Create a list of dictionaries to store the results
        results_list = []

        for player in unique_players:
            # Create a DataFrame with all zeros
            predict_data = pd.DataFrame(0, index=range(1), columns=X.columns)
            # Set the corresponding player's column to 1 for prediction
            predict_data[player] = 1
            # Make a prediction for the player
            misc_fpts_prediction = best_rf_model.predict(predict_data)
            results_list.append({'Player': player, 'MISC FPTS': misc_fpts_prediction[0]})

        # Convert the list of dictionaries into a DataFrame
        results_df = pd.DataFrame(results_list)

        results_df = results_df.sort_values(by='MISC FPTS', ascending=False)
        # Save the results to a CSV file
        file_name = f"predictions/RFweek{num_week}{pos}weight.csv"
        results_df.to_csv(file_name, index=False)

        results_df.head(10)
### END OF CODE CELL ###

### CODE CELL ###
if gpu == True:
    for pos in positions:    
        # Get user input for the week number to predict
        num_week = int(input("Enter the week to predict: "))

        import cudf
        import cuml
        import cupy as cp
        import pandas as pd
        import numpy as np
        from cuml.ensemble import RandomForestRegressor
        from cuml.model_selection import train_test_split, GridSearchCV
        from sklearn.impute import SimpleImputer

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

        # Hyperparameter tuning using GridSearchCV
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        X, y = cp.array(X), cp.array(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Fit the model with the best hyperparameters
        best_rf_model = grid_search.best_estimator_
        best_rf_model.fit(X_train, y_train)

        # Get a list of unique player names after one-hot encoding
        unique_players = X.columns

        # Create a list of dictionaries to store the results
        results_list = []

        for player in unique_players:
            # Create a cuDF DataFrame with all zeros
            predict_data = cudf.DataFrame(np.zeros((1, len(X.columns)), dtype=np.float64))
            # Set the corresponding player's column to 1 for prediction
            predict_data[0, player] = 1
            predict_data = cp.array(predict_data)
            # Make a prediction for the player
            misc_fpts_prediction = best_rf_model.predict(predict_data)
            results_list.append({'Player': player, 'MISC FPTS': misc_fpts_prediction[0]})

        # Convert the list of dictionaries into a cuDF DataFrame
        results_df = cudf.DataFrame(results_list)

        # Save the results to a CSV file
        file_name = f"predictions/RFweek{num_week}{pos}.csv"
        results_df.to_pandas().to_csv(file_name, index=False)

        print(results_df.head(10))
### END OF CODE CELL ###

