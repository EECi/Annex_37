# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:32:50 2023

@author: nm735
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
import os
from  lightgbm import LGBMRegressor


# Tried but unused modules:
# from sklearn.ensemble import RandomForestRegressor # Import the model we are using
# import xgboost as xgb
# from sklearn.multioutput import MultiOutputRegressor
# from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
# from statsmodels.tsa.stattools import adfuller
# from pmdarima import auto_arima


#%% - Functions

def read_in_data_with_1_building(building_id,parent_folder = "analysis", folder="train"):
    
    df_carbon = pd.read_csv(os.path.join("..","..","data",parent_folder,folder,"carbon_intensity.csv"))
    df_pricing = pd.read_csv(os.path.join("..","..","data",parent_folder,folder, "pricing.csv"))
    df_weather = pd.read_csv(os.path.join("..","..","data",parent_folder,folder, "weather.csv"))
    df_B = pd.read_csv(os.path.join("..","..","data",parent_folder,folder,"UCam_Building_" + str(building_id) + ".csv"))

    df_X = pd.concat([df_B.iloc[:,[0,2,1]],df_weather.iloc[:,0:4],df_carbon, df_B.iloc[:,11],df_pricing.iloc[:,0],df_B.rename(columns={df_B.columns[7]:"B0"}).iloc[:,7]],axis=1)
    
    return df_X
    

def read_in_data_with_all_building(parent_folder = "example",folder="train"):
    
    df_carbon = pd.read_csv(os.path.join("..","..","data",parent_folder,folder, "carbon_intensity.csv"))
    df_pricing = pd.read_csv(os.path.join("..","..","data",parent_folder,folder, "pricing.csv"))
    df_weather = pd.read_csv(os.path.join("..","..","data",parent_folder,folder,"weather.csv"))
    buildingFilenamesList = glob.glob(os.path.join("..","..","data",parent_folder,folder, "UCam_Building_*"+ ".csv"))
    df_Bs = []
    for file in buildingFilenamesList:
        df_B = pd.read_csv(file)
        df_Bs.append(df_B)

    df_X = pd.concat([df_Bs[0].iloc[:,[0,2,1]],df_weather.iloc[:,0:4],df_carbon, df_B.iloc[:,[11]],df_pricing.iloc[:,0]],axis=1)

    
    list_df_to_use = []
    list_df_to_use.append(df_X)
    for b_idx, df_B in enumerate(df_Bs):
        list_df_to_use.append(df_B.rename(columns={df_B.columns[7]:"B"+str(b_idx)}).iloc[:,[7]])
    df_X = pd.concat(list_df_to_use, axis=1)
    
    
    return df_X


def create_and_save_LGBM_predictor_iterative(building_id, features_to_incl, feature_to_predict, model_name, L_input=720, save_model = False, test_model = False, plot_results = False):
        
    df_X = read_in_data_with_1_building(building_id,folder="train")
    X_to_use = df_X.iloc[:,features_to_incl].values
    Y_to_use = df_X[[feature_to_predict]].values
    
    X_2D = np.zeros((len(X_to_use)-L_input,L_input))
    Y_1D = np.zeros((len(Y_to_use)-L_input,))
    for i in range (len(X_to_use)-L_input):
        X_2D[i,:] = X_to_use[i:i+L_input,:].reshape(-1)
        Y_1D[i] = Y_to_use[i+L_input,:]

    # Create & train predictor    
    predictive_model = LGBMRegressor(verbose=-1)
    predictive_model.fit(X_2D,Y_1D)

    # Save predictor
    if save_model:
        pickle.dump(predictive_model,open("pre_trained/" + model_name, "wb"), -1)
    
    ## TESTING
    if test_model:
        df_X_t = read_in_data_with_1_building(building_id,folder="validate")
        X_t_to_use = df_X_t.iloc[:,features_to_incl].values
        Y_t_to_use = df_X_t[[feature_to_predict]].values
        
        X_t_2D = np.zeros((len(X_t_to_use)-L_input,L_input))
        Y_t_1D = np.zeros((len(Y_t_to_use)-L_input,))
        for i in range (len(X_t_to_use)-L_input):
            X_t_2D[i,:] = X_t_to_use[i:i+L_input,:].reshape(-1)
            Y_t_1D[i] = Y_t_to_use[i+L_input,:]
        
        # Predict test data
        predictions = predictive_model.predict(X_t_2D)
        # Calculate the absolute errors
        errors = abs(predictions - Y_t_1D)
        mean_errors = np.mean(errors, axis=0)
        # Print out the mean absolute error (mae)
        print(feature_to_predict)
        print(mean_errors)
            
        # Plot - very rough to check
        if (plot_results):
            # for c_id, col in enumerate(df_Y_t_to_predict.columns):
            plt.figure() #1
            plt.title(feature_to_predict)
            plt.scatter(predictions, Y_t_1D)
            plt.show()
            
            plt.figure() #2
            plt.title(feature_to_predict)
            plt.plot(predictions, label="predictions")
            plt.plot(Y_t_1D, label = "actual")
            plt.legend()
            plt.show()
    
            plt.figure() #3
            plt.title(feature_to_predict)
            plt.plot(predictions - Y_t_1D, label="error")
            plt.legend()
            plt.show()
    else:
        print(feature_to_predict)
        
        
def create_and_save_LGBM_predictor_non_iterative(building_id, features_to_incl, feature_to_predict, model_name, L_input=720, tau=48, save_model = False, test_model = False, plot_results = False):
        
    df_X = read_in_data_with_1_building(building_id,folder="train")
    X_to_use = df_X.iloc[:,features_to_incl].values
    Y_to_use = df_X[[feature_to_predict]].values

    X_2D = np.zeros((len(X_to_use)-L_input-tau,L_input*len(features_to_incl)))
    Y_1D = np.zeros((len(Y_to_use)-L_input-tau,))
    for i in range(len(X_to_use)-L_input-tau):
        X_2D[i,:] = X_to_use[i:i+L_input,:].ravel(order='F')
        Y_1D[i] = Y_to_use[i+L_input+tau,:]

    # Create & train predictor    
    predictive_model = LGBMRegressor(verbose=-1)
    predictive_model.fit(X_2D,Y_1D)

    # Save predictor
    if save_model:
        pickle.dump(predictive_model,open("pre_trained/" + model_name, "wb"), -1)
    
    ## TESTING
    if test_model:
        df_X_t = read_in_data_with_1_building(building_id,folder="validate")
        X_t_to_use = df_X_t.iloc[:,features_to_incl].values
        Y_t_to_use = df_X_t[[feature_to_predict]].values

        X_t_2D = np.zeros((len(X_t_to_use)-L_input,L_input*len(features_to_incl)))
        Y_t_1D = np.zeros((len(Y_t_to_use)-L_input,))
        for i in range (len(X_t_to_use)-L_input):
            X_t_2D[i,:] = X_t_to_use[i:i+L_input,:].ravel(order='F')
            Y_t_1D[i] = Y_t_to_use[i+L_input,:]
        # Predict test data
        predictions = predictive_model.predict(X_t_2D)
        # Calculate the absolute errors
        errors = abs(predictions - Y_t_1D)
        mean_errors = np.mean(errors, axis=0)

        print(feature_to_predict)
        print(mean_errors)
            
        # Plot - very rough to check
        if (plot_results):

            plt.figure() #1
            plt.title(feature_to_predict)
            plt.scatter(predictions, Y_t_1D)
            plt.show()
            
            plt.figure() #2
            plt.title(feature_to_predict)
            plt.plot(predictions, label="predictions")
            plt.plot(Y_t_1D, label = "actual")
            plt.legend()
            plt.show()
    
            plt.figure() #3
            plt.title(feature_to_predict)
            plt.plot(predictions - Y_t_1D, label="error")
            plt.legend()
            plt.show()
    else:
        print(feature_to_predict)



# %% Create predictors for all needs
# buildings =  [5,11,14,16,24,29] # example set
buildings =  [0,3,9,11,12,15,16,25,26,32,38,44,45,48,49] # analysis set
tau = 48
L_input = 720
# features_to_incl = [0,1,2,3,4,5,6,7,8,9,10]
features_B = [10]
features_c = [7]
features_cost = [9]


iterative_bool = True

test_model_bool = True
save_model_bool = True
plot_bool = False

version_name = "v1"

    # Create and train predictive models - two different approaches
if iterative_bool:
    features_solar = [8] # predict only solar - no control inputs because of iterative approach        
    base_model_name = "LGBM_iterative_t" + str(1) + "_singleFeature"

    for building_id in buildings:
        model_name = base_model_name + "_B" + str(building_id)+"_" + version_name
        create_and_save_LGBM_predictor_iterative(building_id, features_B, "B0", model_name, L_input=L_input, save_model = save_model_bool, test_model = test_model_bool, plot_results = plot_bool)    
    create_and_save_LGBM_predictor_iterative(buildings[0], features_c, 'kg_CO2/kWh', base_model_name + "_carbon"+"_" + version_name, L_input=L_input, save_model = save_model_bool, test_model = test_model_bool, plot_results = plot_bool)
    create_and_save_LGBM_predictor_iterative(buildings[0], features_solar, 'Solar Generation [W/kW]', base_model_name + "_solar"+"_" + version_name, L_input=L_input, save_model = save_model_bool, test_model = test_model_bool, plot_results = plot_bool)
    create_and_save_LGBM_predictor_iterative(buildings[0], features_cost, 'Electricity Pricing [£/kWh]', base_model_name + "_price"+"_" + version_name, L_input=L_input, save_model = save_model_bool, test_model = test_model_bool, plot_results = plot_bool)
    
else:    
    features_solar = [8,5,6] 
    base_model_name = "LGBM_non_iterative_t" + str(1) + "_singleFeature"

    for building_id in buildings:
        model_name = base_model_name + "_B" + str(building_id)+"_" + version_name
        create_and_save_LGBM_predictor_non_iterative(building_id, features_B, "B0", model_name, L_input=L_input, tau=tau, save_model = save_model_bool, test_model = test_model_bool, plot_results = plot_bool)   
    create_and_save_LGBM_predictor_non_iterative(buildings[0], features_c, 'kg_CO2/kWh', base_model_name + "_carbon"+"_" + version_name, L_input=L_input, tau=tau, save_model = save_model_bool, test_model = test_model_bool, plot_results = plot_bool)
    create_and_save_LGBM_predictor_non_iterative(buildings[0], features_solar, 'Solar Generation [W/kW]', base_model_name + "_solar"+"_" + version_name, L_input=L_input, tau=tau, save_model = save_model_bool, test_model = test_model_bool, plot_results = True)
    create_and_save_LGBM_predictor_non_iterative(buildings[0], features_cost, 'Electricity Pricing [£/kWh]', base_model_name + "_price"+"_" + version_name, L_input=L_input, tau=tau, save_model = save_model_bool, test_model = test_model_bool, plot_results = plot_bool)




