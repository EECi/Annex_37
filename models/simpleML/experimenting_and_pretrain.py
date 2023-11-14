# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:32:50 2023

@author: nm735
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.ensemble import RandomForestRegressor # Import the model we are using
import pickle
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.stattools import adfuller

from pmdarima import auto_arima

#%% - Functions

def read_in_data_with_1_building(building_id,folder="train"):
    
    df_carbon = pd.read_csv("data/example/" + folder + "/carbon_intensity.csv")
    df_pricing = pd.read_csv("data/example/" + folder + "/pricing.csv")
    df_weather = pd.read_csv("data/example/" + folder + "/weather.csv")
    df_B = pd.read_csv("data/example/" + folder + "/UCam_Building_" + str(building_id) + ".csv")
    # df_B1 = pd.read_csv("data/example/train/UCam_Building_5.csv")
    # df_B2 = pd.read_csv("data/example/train/UCam_Building_11.csv")
    # df_B3 = pd.read_csv("data/example/train/UCam_Building_14.csv")
    # df_B4 = pd.read_csv("data/example/train/UCam_Building_16.csv")
    # df_B5 = pd.read_csv("data/example/train/UCam_Building_24.csv")
    # df_B6 = pd.read_csv("data/example/train/UCam_Building_29.csv")
    # df_Bs = [df_B1, df_B2, df_B3, df_B4, df_B5, df_B6]

    # df_X = pd.concat([df_carbon,df_pricing.iloc[:,0],df_weather.iloc[:,0:4], df_B.rename(columns={df_B.columns[7]:"B0"}).iloc[:,[0,1,2,7,11]]],axis=1)
    df_X = pd.concat([df_B.iloc[:,[0,2,1]],df_weather.iloc[:,0:4],df_carbon, df_B.iloc[:,11],df_pricing.iloc[:,0],df_B.rename(columns={df_B.columns[7]:"B0"}).iloc[:,7]],axis=1)
    
    
    
    # list_df_to_use = []
    # list_df_to_use.append(df_X)
    # list_df_to_use.append(df_B1.iloc[:,[0,1,2,11]]) # assuming solar generation (11) same for all
    # for b_idx, df_B in enumerate(df_Bs):
    #     list_df_to_use.append(df_B.rename(columns={df_B1.columns[7]:"B"+str(b_idx)}).iloc[:,[7]])
    # df_X = pd.concat(list_df_to_use, axis=1)
    
    # df_X_to_use = df_X.iloc[:-1,]
    # df_Y_all =  df_X.iloc[1:,]
    
    return df_X
    
    # df_Y_to_predict = df_Y_all["B0"]
    

def read_in_data_with_all_building(folder="train"):
    
    df_carbon = pd.read_csv("data/example/" + folder + "/carbon_intensity.csv")
    df_pricing = pd.read_csv("data/example/" + folder + "/pricing.csv")
    df_weather = pd.read_csv("data/example/" + folder + "/weather.csv")
    buildingFilenamesList = glob.glob("data/example/" + folder + "/UCam_Building_*"+ ".csv")
    df_Bs = []
    for file in buildingFilenamesList:
        df_B = pd.read_csv(file)
        df_Bs.append(df_B)

    # df_X = pd.concat([df_carbon,df_pricing.iloc[:,0],df_weather.iloc[:,0:4], df_Bs[0].iloc[:,[0,1,2,11]]],axis=1)
    df_X = pd.concat([df_Bs[0].iloc[:,[0,2,1]],df_weather.iloc[:,0:4],df_carbon, df_B.iloc[:,[11]],df_pricing.iloc[:,0]],axis=1)

    
    list_df_to_use = []
    list_df_to_use.append(df_X)
    for b_idx, df_B in enumerate(df_Bs):
        list_df_to_use.append(df_B.rename(columns={df_B.columns[7]:"B"+str(b_idx)}).iloc[:,[7]])
    df_X = pd.concat(list_df_to_use, axis=1)
    
    # df_X_to_use = df_X.iloc[:-1,]
    # df_Y_all =  df_X.iloc[1:,]
    
    return df_X

def add_cummulative_load(df, column_name, increments):
    for incr in increments:
        df[column_name + "_cum_" + str(incr)] = df[column_name].rolling(incr, min_periods=1).sum()
    df = df.iloc[max(increments):,]
    return df

def get_random_forest_feature_importance(df_X, df_Y):
    # Instantiate model with X decision trees
    rf = RandomForestRegressor(n_estimators = 10, random_state = 42)
    # Train the model on training data
    rf.fit(df_X.values, df_Y.values)
    
    pltdata = rf.feature_importances_
    fig, ax = plt.subplots()
    labels = df_X.columns
    plt.xticks(range(len(pltdata)), labels, rotation='vertical')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title("Predicting: " + df_Y.name)    
    ax.bar(range(len(pltdata)), pltdata,log=True)
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.show()
    
    return(rf.feature_importances_)

def create_and_save_RF_predictor(building_id, tau, features_to_incl, feature_to_predict, model_name, test_model = False, plot_results = False):
    
    df_X = read_in_data_with_1_building(building_id,folder="train")
    df_X_to_use = df_X.iloc[:-1*tau,features_to_incl]

    df_Y_to_predict = pd.DataFrame()
    for i in range(tau):
        df_Y_to_predict["T_" + str(i)] = df_X[feature_to_predict].shift(-1*(i+1))
    df_Y_to_predict = df_Y_to_predict.iloc[:-1*tau,]

    # Instantiate model with X decision trees
    rf = RandomForestRegressor(n_estimators = 200, random_state = 42)
    # Train the model on training data
    rf.fit(df_X_to_use.values, df_Y_to_predict.values)
    # print(rf.feature_importances_)
    
    pickle.dump(rf,open("models/" + model_name, "wb"), -1)
    
    ## TESTING
    if test_model:
        df_X_t = read_in_data_with_1_building(building_id,folder="test")
        df_X_t_to_use = df_X_t.iloc[:-1*tau,features_to_incl] #[7,9] [0,1,2,3,7,9]
        df_Y_t_to_predict = pd.DataFrame()
        for i in range(tau):
            df_Y_t_to_predict["T_" + str(i)] = df_X_t[df_X_t.columns[-1]].shift(-1*(i+1))
        df_Y_t_to_predict = df_Y_t_to_predict.iloc[:-1*tau,]
    
        # Use the forest's predict method on the test data
        predictions = rf.predict(df_X_t_to_use.values)
        # Calculate the absolute errors
        errors = abs(predictions - df_Y_t_to_predict.values)
        mean_errors = np.mean(errors, axis=0)
        # Print out the mean absolute error (mae)
        for error in mean_errors:
            print('Mean Absolute Error:', round(error, 2), 'W/kW.')
            
    # Plot - very rough
    if (plot_results):
        plt.figure(1)
        plt.scatter(predictions, df_Y_t_to_predict.values)
        plt.show()
        
        plt.figure(2)
        plt.plot(predictions, label="predictions")
        plt.plot(df_Y_t_to_predict.values, label = "actual")
        plt.legend()
        plt.show()

        plt.figure(3)
        plt.plot(predictions - df_Y_t_to_predict.values, label="error")
        plt.legend()
        plt.show()

def create_and_save_RF_predictor_withHistory(building_id, tau, features_to_incl, feature_to_predict, model_name, cumulatives_to_use, test_model = False, plot_results = False):
        
    df_X = read_in_data_with_1_building(building_id,folder="train")
    df_X_to_use = df_X.iloc[:-1*tau,features_to_incl]

    df_Y_to_predict = pd.DataFrame()
    for i in range(tau):
        df_Y_to_predict["T_" + str(i)] = df_X[feature_to_predict].shift(-1*(i+1))
    df_Y_to_predict = df_Y_to_predict.iloc[:-1*tau,]
    
    if cumulatives_to_use: # Not empty
        df_X_to_use = add_cummulative_load(df_X_to_use.copy(), feature_to_predict, cumulatives_to_use)
        df_Y_to_predict =  df_Y_to_predict.iloc[max(cumulatives_to_use):,]


    # Instantiate model with X decision trees
    rf = RandomForestRegressor(n_estimators = 200, random_state = 42)
    # Train the model on training data
    rf.fit(df_X_to_use.values, df_Y_to_predict.values)
    # print(rf.feature_importances_)
    
    pickle.dump(rf,open("models/" + model_name, "wb"), -1)
    
    ## TESTING
    if test_model:
        df_X_t = read_in_data_with_1_building(building_id,folder="test")
        df_X_t_to_use = df_X_t.iloc[:-1*tau,features_to_incl] #[7,9] [0,1,2,3,7,9]
        df_Y_t_to_predict = pd.DataFrame()
        for i in range(tau):
            df_Y_t_to_predict["T_" + str(i)] = df_X_t[feature_to_predict].shift(-1*(i+1))
        df_Y_t_to_predict = df_Y_t_to_predict.iloc[:-1*tau,]
        
        if cumulatives_to_use: # Not empty
            df_X_t_to_use = add_cummulative_load(df_X_t_to_use.copy(), feature_to_predict, cumulatives_to_use)
            df_Y_t_to_predict =  df_Y_t_to_predict.iloc[max(cumulatives_to_use):,]

    
        # Use the forest's predict method on the test data
        predictions = rf.predict(df_X_t_to_use.values)
        # Calculate the absolute errors
        errors = abs(predictions - df_Y_t_to_predict.values)
        mean_errors = np.mean(errors, axis=0)
        # Print out the mean absolute error (mae)
        print(feature_to_predict)
        for error in mean_errors:
            print('Mean Absolute Error:', round(error, 2), '.')
            
    # Plot - very rough
    if (plot_results):
        plt.figure(1)
        plt.scatter(predictions, df_Y_t_to_predict.values)
        plt.show()
        
        plt.figure(2)
        plt.plot(predictions, label="predictions")
        plt.plot(df_Y_t_to_predict.values, label = "actual")
        plt.legend()
        plt.show()

        plt.figure(3)
        plt.plot(predictions - df_Y_t_to_predict.values, label="error")
        plt.legend()
        plt.show()
        
   
def create_XGBoost(X, Y):
    # Instantiate model
    my_xgb = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror'))
    # Train the model on training data
    my_xgb.fit(X, Y)
    return my_xgb

def test_model(building_id, tau, features_to_incl, feature_to_predict, model):
    df_X_t = read_in_data_with_1_building(building_id,folder="test")
    df_X_t_to_use = df_X_t.iloc[:-1*tau,features_to_incl] #[7,9] [0,1,2,3,7,9]
    df_Y_t_to_predict = pd.DataFrame()
    for i in range(tau):
        df_Y_t_to_predict["T_" + str(i)] = df_X_t[feature_to_predict].shift(-1*(i+1))
    df_Y_t_to_predict = df_Y_t_to_predict.iloc[:-1*tau,]
    
    if cumulatives_to_use: # Not empty
        df_X_t_to_use = add_cummulative_load(df_X_t_to_use.copy(), feature_to_predict, cumulatives_to_use)
        df_Y_t_to_predict =  df_Y_t_to_predict.iloc[max(cumulatives_to_use):,]


    # Use the forest's predict method on the test data
    predictions = model.predict(df_X_t_to_use.values)
    # Calculate the absolute errors
    errors = abs(predictions - df_Y_t_to_predict.values)
    mean_errors = np.mean(errors, axis=0)
    # Print out the mean absolute error (mae)
    print(feature_to_predict)
    for error in mean_errors:
        print('Mean Absolute Error:', round(error, 2), '.')

   
def create_and_save_XGBoost_predictor_withHistory(building_id, tau, features_to_incl, feature_to_predict, model_name, cumulatives_to_use, save_model = False, test_model = False, plot_results = False):
        
    df_X = read_in_data_with_1_building(building_id,folder="train")
    df_X_to_use = df_X.iloc[:-1*tau,features_to_incl]

    df_Y_to_predict = pd.DataFrame()
    for i in range(tau):
        df_Y_to_predict["T_" + str(i)] = df_X[feature_to_predict].shift(-1*(i+1))
    df_Y_to_predict = df_Y_to_predict.iloc[:-1*tau,]
    
    if cumulatives_to_use: # Not empty
        df_X_to_use = add_cummulative_load(df_X_to_use.copy(), feature_to_predict, cumulatives_to_use)
        df_Y_to_predict =  df_Y_to_predict.iloc[max(cumulatives_to_use):,]

    # Create & train predictor    
    my_xgb  = create_XGBoost(df_X_to_use.values, df_Y_to_predict.values)

    # Save predictor
    if save_model:
        pickle.dump(my_xgb,open("models/" + model_name, "wb"), -1)
    
    ## TESTING
    if test_model:
        df_X_t = read_in_data_with_1_building(building_id,folder="test")
        df_X_t_to_use = df_X_t.iloc[:-1*tau,features_to_incl] #[7,9] [0,1,2,3,7,9]
        df_Y_t_to_predict = pd.DataFrame()
        for i in range(tau):
            df_Y_t_to_predict["T_" + str(i)] = df_X_t[feature_to_predict].shift(-1*(i+1))
        df_Y_t_to_predict = df_Y_t_to_predict.iloc[:-1*tau,]
        
        if cumulatives_to_use: # Not empty
            df_X_t_to_use = add_cummulative_load(df_X_t_to_use.copy(), feature_to_predict, cumulatives_to_use)
            df_Y_t_to_predict =  df_Y_t_to_predict.iloc[max(cumulatives_to_use):,]

    
        # Use the forest's predict method on the test data
        predictions = my_xgb.predict(df_X_t_to_use.values)
        # Calculate the absolute errors
        errors = abs(predictions - df_Y_t_to_predict.values)
        mean_errors = np.mean(errors, axis=0)
        # Print out the mean absolute error (mae)
        print(feature_to_predict)
        for error in mean_errors:
            print('Mean Absolute Error:', round(error, 2), '.')
            
        # Plot - very rough
        if (plot_results):
            for c_id, col in enumerate(df_Y_t_to_predict.columns):
                plt.figure() #1
                plt.title(col)
                plt.scatter(predictions[:,c_id], df_Y_t_to_predict.values[:,c_id])
                plt.show()
                
                plt.figure() #2
                plt.title(col)
                plt.plot(predictions[:,c_id], label="predictions")
                plt.plot(df_Y_t_to_predict.values[:,c_id], label = "actual")
                plt.legend()
                plt.show()
        
                plt.figure() #3
                plt.title(col)
                plt.plot(predictions[:,c_id] - df_Y_t_to_predict.values[:,c_id], label="error")
                plt.legend()
                plt.show()
    else:
        print(feature_to_predict)

def test_ARIMA():    
    df_X = read_in_data_with_all_building()
    
    predict_labels = ["Solar Generation [W/kW]","kg_CO2/kWh", "Electricity Pricing [£/kWh]", "B0"]
    
    for label in predict_labels:
    
        data = df_X[label][-720:]
        # plot_pacf(data.diff().dropna());
        # plot_acf(data.diff().dropna());
        
        # p_value = adfuller(data, autolag = 'AIC')[1]
        # count_d = 0
        # while(p_value>0.05):
        #     data = data.diff().dropna()
        #     p_value = adfuller(data, autolag = 'AIC')[1]
        #     count_d +=1
        # print(count_d)
        
        print(label)
        data.plot()
        # auto_arima(data, start_p=1,max_p=6, start_q=1, max_q=6, seasonal=True, trace = True).summary()
        auto_arima(data, trace = True).summary() #m=24, seasonal=True,

    # data = df_X["Solar Generation [W/kW]"][-720:]
    # data_exog = df_X[["Diffuse Solar Radiation [W/m2]","Direct Solar Radiation [W/m2]"]][-720:]
    # auto_arima(data,  m=24, seasonal=True, trace = True,exogenous = data_exog).summary()
    



# %% Create predictors for all needs
buildings =  [5,11,14,16,24,29]
tau = 12
features_to_incl = [0,1,2,3,4,5,6,7,8,9,10]
# features_to_incl = []
features_solar = features_to_incl #[8]
features_B = features_to_incl #[10]
features_c = features_to_incl #[7]
features_cost = features_to_incl #[9]

# # Attempt 1 - simple
# for building_id in buildings:
#     model_name = "Random_Forest_dummy_t12_B" + str(building_id)
#     feature_to_predict = "B0"
#     create_and_save_RF_predictor(building_id, tau, features_to_incl, feature_to_predict, model_name, test_model = False, plot_results = False)

# create_and_save_RF_predictor(buildings[0], tau, features_to_incl, 'kg_CO2/kWh', "Random_Forest_dummy_t12_carbon", test_model = False, plot_results = False)
# create_and_save_RF_predictor(buildings[0], tau, features_to_incl, 'Solar Generation [W/kW]', "Random_Forest_dummy_t12_solar", test_model = False, plot_results = False)
# create_and_save_RF_predictor(buildings[0], tau, features_to_incl, 'Electricity Pricing [£/kWh]', "Random_Forest_dummy_t12_electricity", test_model = False, plot_results = False)


# Attempt 2 - with History
test_model_bool = True
save_model_bool = True

cumulatives_to_use = [12,24,48,7*24]



base_model_name = "XGBoost_dummy_t" + str(tau) + "_singleFeature"
H_name = "H1"

for building_id in buildings:
    # model_name = "Random_Forest_dummy_t12_B" + str(building_id)+"_H1"
    # create_and_save_RF_predictor_withHistory(building_id, tau, features_to_incl, "B0", model_name, cumulatives_to_use, test_model = test_model_bool, plot_results = False)
    model_name = base_model_name + "_B" + str(building_id)+"_" + H_name
    create_and_save_XGBoost_predictor_withHistory(building_id, tau, features_B, "B0", model_name, cumulatives_to_use, save_model = save_model_bool, test_model = test_model_bool, plot_results = False)
        
# create_and_save_RF_predictor_withHistory(buildings[0], tau, features_to_incl, 'kg_CO2/kWh', "Random_Forest_dummy_t12_carbon_H1", cumulatives_to_use, test_model = test_model_bool, plot_results = False)
# create_and_save_RF_predictor_withHistory(buildings[0], tau, features_to_incl, 'Solar Generation [W/kW]', "Random_Forest_dummy_t12_solar_H1", cumulatives_to_use, test_model = test_model_bool, plot_results = False)
# create_and_save_RF_predictor_withHistory(buildings[0], tau, features_to_incl, 'Electricity Pricing [£/kWh]', "Random_Forest_dummy_t12_electricity_H1", cumulatives_to_use, test_model = test_model_bool, plot_results = True)

create_and_save_XGBoost_predictor_withHistory(buildings[0], tau, features_c, 'kg_CO2/kWh', base_model_name + "_carbon"+"_" + H_name, cumulatives_to_use, save_model = save_model_bool, test_model = test_model_bool, plot_results = False)
create_and_save_XGBoost_predictor_withHistory(buildings[0], tau, features_solar, 'Solar Generation [W/kW]', base_model_name + "_solar"+"_" + H_name, cumulatives_to_use, save_model = save_model_bool, test_model = test_model_bool, plot_results = False)
create_and_save_XGBoost_predictor_withHistory(buildings[0], tau, features_cost, 'Electricity Pricing [£/kWh]', base_model_name + "_electricity"+"_" + H_name, cumulatives_to_use, save_model = save_model_bool, test_model = test_model_bool, plot_results = False)






#%% Feature importance for each feature to predict

# # Carbon emissions + Electricity
# df_X_in = read_in_data_with_all_building(folder="train")
# df_X_all = df_X_in.iloc[:-1,]
# df_Y_all = df_X_in.iloc[1:,]

# # Importance plotting
# # get_random_forest_feature_importance(df_X_all, df_Y_all.iloc[:,0])
# # get_random_forest_feature_importance(df_X_all, df_Y_all.iloc[:,1])
# # get_random_forest_feature_importance(df_X_all, df_Y_all.iloc[:,9])
# # get_random_forest_feature_importance(df_X_all, df_Y_all.iloc[:,10])

#%% Random Forests Solar
# building_id = 5
# plot_results = False
# # features_to_incl = [4,5,7,10]
# features_to_incl = [0,2,5,6,8]


# df_X = read_in_data_with_1_building(building_id,folder="train")
# df_X_to_use = df_X.iloc[:-1,features_to_incl]
# df_Y_all =  df_X.iloc[1:,]
# df_Y_to_predict = df_Y_all[df_Y_all.columns[-1]]

# # Instantiate model with X decision trees
# rf = RandomForestRegressor(n_estimators = 200, random_state = 42)
# # Train the model on training data
# rf.fit(df_X_to_use.values, df_Y_to_predict.values)
# print(rf.feature_importances_)

# ## TESTING
# df_X_t = read_in_data_with_1_building(building_id,folder="test")
# df_X_t_to_use = df_X_t.iloc[:-1,features_to_incl] #[7,9] [0,1,2,3,7,9]
# df_Y_t_all =  df_X_t.iloc[1:,]
# df_Y_t_to_predict = df_Y_t_all[df_Y_t_all.columns[-1]]

# # Use the forest's predict method on the test data
# predictions = rf.predict(df_X_t_to_use.values)
# # Calculate the absolute errors
# errors = abs(predictions - df_Y_t_to_predict.values)
# # Print out the mean absolute error (mae)
# print('Mean Absolute Error:', round(np.mean(errors), 2), 'W/kW.')

# # Plot - very rough
# if (plot_results):
#     plt.figure(1)
#     plt.scatter(predictions, df_Y_t_to_predict.values)
#     plt.show()
    
#     plt.figure(2)
#     plt.plot(predictions, label="predictions")
#     plt.plot(df_Y_t_to_predict.values, label = "actual")
#     plt.legend()
#     plt.show()
    
#     plt.figure(3)
#     plt.plot(predictions - df_Y_t_to_predict.values, label="error")
#     plt.legend()
#     plt.show()



#%% Random Forests Solar - X time steps
# building_id = 5
# plot_results = False
# features_to_incl = [0,1,2,3,4,5,6,7,8,9,10] #[4,5,7,10]
# time_steps = 12

# df_X = read_in_data_with_1_building(building_id,folder="train")
# df_X_to_use = df_X.iloc[:-1*time_steps,features_to_incl]

# df_Y_to_predict = pd.DataFrame()
# for i in range(time_steps):
#     df_Y_to_predict["T_" + str(i)] = df_X[df_X.columns[-1]].shift(-1*(i+1))
# df_Y_to_predict = df_Y_to_predict.iloc[:-1*time_steps,]

# # Instantiate model with X decision trees
# rf = RandomForestRegressor(n_estimators = 200, random_state = 42)
# # Train the model on training data
# rf.fit(df_X_to_use.values, df_Y_to_predict.values)
# print(rf.feature_importances_)

# ## TESTING
# df_X_t = read_in_data_with_1_building(building_id,folder="test")
# df_X_t_to_use = df_X_t.iloc[:-1*time_steps,features_to_incl] #[7,9] [0,1,2,3,7,9]
# df_Y_t_to_predict = pd.DataFrame()
# for i in range(time_steps):
#     df_Y_t_to_predict["T_" + str(i)] = df_X_t[df_X_t.columns[-1]].shift(-1*(i+1))
# df_Y_t_to_predict = df_Y_t_to_predict.iloc[:-1*time_steps,]

# # Use the forest's predict method on the test data
# predictions = rf.predict(df_X_t_to_use.values)
# # Calculate the absolute errors
# errors = abs(predictions - df_Y_t_to_predict.values)
# mean_errors = np.mean(errors, axis=0)
# # Print out the mean absolute error (mae)
# for error in mean_errors:
#     print('Mean Absolute Error:', round(error, 2), 'W/kW.')

# pickle.dump(rf,open("models/Random_Forest_dummy_Solar", "wb"))

# # Plot - very rough
# if (plot_results):
#     plt.figure(1)
#     plt.scatter(predictions, df_Y_t_to_predict.values)
#     plt.show()
    
#     plt.figure(2)
#     plt.plot(predictions, label="predictions")
#     plt.plot(df_Y_t_to_predict.values, label = "actual")
#     plt.legend()
#     plt.show()

#     plt.figure(3)
#     plt.plot(predictions - df_Y_t_to_predict.values, label="error")
#     plt.legend()
#     plt.show()




#%% Random Forests Single Building
# building_id = 5
# cumulatives_to_use = []# [24*7] #[] #
# plot_results = False
# features_to_incl = [0,1,2,3,4,5,6,7,8,9] #[0,1,2,3,4,7,8,9] #[7,9] #
# ## TRAINING

# df_X = read_in_data_with_1_building(building_id,folder="train")
# df_X_to_use = df_X.iloc[:-1,features_to_incl] #[7,9] [7,9] [0,1,2,3,7,9]
# if cumulatives_to_use: # Not empty
#     df_X_to_use = add_cummulative_load(df_X_to_use.copy(), "B0", cumulatives_to_use)
#     df_Y_all =  df_X.iloc[1 + cumulatives_to_use[-1]:,]
# else:
#     df_Y_all =  df_X.iloc[1:,]

# # changing day to weekday or not
# # df_X_to_use.loc[df_X_to_use["Day Type"]==1,"Day Type"]=0
# # df_X_to_use.loc[df_X_to_use["Day Type"]>=1,"Day Type"]=1
    
# df_Y_to_predict = df_Y_all["B0"]


# # Instantiate model with X decision trees
# rf = RandomForestRegressor(n_estimators = 200, random_state = 42)
# # Train the model on training data
# rf.fit(df_X_to_use.values, df_Y_to_predict.values)
# rf.feature_importances_


# ## TESTING
# df_X_t = read_in_data_with_1_building(building_id,folder="test")
# df_X_t_to_use = df_X_t.iloc[:-1,features_to_incl] #[7,9] [0,1,2,3,7,9]
# if cumulatives_to_use: # Not empty
#     df_X_t_to_use = add_cummulative_load(df_X_t_to_use.copy(), "B0", cumulatives_to_use)
#     df_Y_t_all =  df_X_t.iloc[1 + cumulatives_to_use[-1]:,]
# else:
#     df_Y_t_all =  df_X_t.iloc[1:,]
# # df_X_t_to_use.loc[df_X_t_to_use["Day Type"]==1,"Day Type"]=0
# # df_X_t_to_use.loc[df_X_t_to_use["Day Type"]>=1,"Day Type"]=1

# df_Y_t_to_predict = df_Y_t_all["B0"]

# # Use the forest's predict method on the test data
# predictions = rf.predict(df_X_t_to_use.values)
# # Calculate the absolute errors
# errors = abs(predictions - df_Y_t_to_predict.values)
# # Print out the mean absolute error (mae)
# print('Mean Absolute Error:', round(np.mean(errors), 2), 'kWh.')

# # Plot - very rough
# if (plot_results):
#     plt.figure(1)
#     plt.scatter(predictions, df_Y_t_to_predict.values)
#     plt.show()
    
#     plt.figure(2)
#     plt.plot(predictions, label="predictions")
#     plt.plot(df_Y_t_to_predict.values, label = "actual")
#     plt.legend()
#     plt.show()



