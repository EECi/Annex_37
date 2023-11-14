"""
Implementation of your prediction method.

The Predictor component of the Linear MPC controller is implemented
as a class.
This class must have the following methods:
    - __init__(self, ...), which initialises the Predictor object and
        performs any initial setup you might want to do.
    - compute_forecast(observation), which executes your prediction method,
        creating timeseries forecasts for [building electrical loads,
        building solar pv generation powers, grid electricity price, grid
        carbon intensity] given the current observation.

You may wish to implement additional methods to make your model code neater.
"""

import numpy as np
import pickle
import pandas as pd
import glob
import xgboost


class XGBPreTPredictor:

    # TODO: take building list as input - as other classes
    def __init__(self, N: int, tau: int, str_model_version = "_H1"):
        """Initialise Prediction object and perform setup.
        
        Args:
            N (int): number of buildings in model, hence number of buildings
                requiring forecasts.
            tau (int): length of planning horizon (number of time instances
                into the future to forecast).
                Note: with some adjustment of the codebase variable length
                planning horizons can be implemented.
        """

        self.num_buildings = N
        self.tau = tau

        # Load in pre-computed prediction model.
        # ====================================================================
        # insert your loading code here
        # ====================================================================

        # Create buffer/tracking attributes
        self.prev_observations = None
        self.buffer = {'key': []}
        # ====================================================================


       # TODO: check folders and paths
       
        self.pre_trained_models_Bs = {}
        # str_model_version = "_H1"
        str_model_type = "XGBoost_dummy_t"+ str(tau)+"_singleFeature"
        
        # TODO: train models for analysis instead of example
        self.pre_trained_models_solar = pickle.load(open("pre_trained/"+str_model_type +"_solar" + str_model_version, "rb"))
        self.pre_trained_models_Bs[0] = pickle.load(open("pre_trained/"+str_model_type +"_B5" + str_model_version, "rb"))
        self.pre_trained_models_Bs[1] = pickle.load(open("pre_trained/"+str_model_type +"_B11"+ str_model_version, "rb"))
        self.pre_trained_models_Bs[2]= pickle.load(open("pre_trained/"+str_model_type +"_B14"+ str_model_version, "rb"))
        self.pre_trained_models_Bs[3] = pickle.load(open("pre_trained/"+str_model_type +"_B16"+ str_model_version, "rb"))
        self.pre_trained_models_Bs[4] = pickle.load(open("pre_trained/"+str_model_type +"_B24"+ str_model_version, "rb"))
        self.pre_trained_models_Bs[5] = pickle.load(open("pre_trained/"+str_model_type +"_B29"+ str_model_version, "rb"))
        self.pre_trained_models_carbon = pickle.load(open("pre_trained/"+str_model_type +"_carbon"+ str_model_version, "rb"))
        self.pre_trained_models_elec = pickle.load(open("pre_trained/"+str_model_type +"_electricity"+ str_model_version, "rb"))
        
        self.model_version = str_model_version
        # ====================================================================
        if str_model_version in ["_H1", "_H2", "_H3"]:
            self.increments = [12,24,48,7*24]
            self.set_initial_cummulative_arrays()
            self.set_cummulative_values()
            
        else:
            self.increments = []
            self.cum_arrays = {'loads': None, 'pv_gens': None, 'price': None, 'carbon': None}
            self.cum_values = {'loads': None, 'pv_gens': None, 'price': None, 'carbon': None}

        # ====================================================================
 
    def read_data_1_building(self,building_id, folder = "analysis", data_set = "validate"):

        df_carbon = pd.read_csv("data/" + folder + "/" +data_set + "/carbon_intensity.csv")
        df_pricing = pd.read_csv("data/" + folder + "/" +data_set + "/pricing.csv")
        df_weather = pd.read_csv("data/" + folder + "/" +data_set + "/weather.csv")
        df_B = pd.read_csv("data/" + folder + "/" +data_set + "/UCam_Building_" + str(building_id) + ".csv")
        df_X = pd.concat([df_B.iloc[:,[0,2,1]],df_weather.iloc[:,0:4],df_carbon, df_B.iloc[:,11],df_pricing.iloc[:,0],df_B.rename(columns={df_B.columns[7]:"B0"}).iloc[:,7]],axis=1)        
        return df_X
 
    def set_initial_cummulative_arrays(self, folder = "analysis", data_set = "validate"):

        df_carbon = pd.read_csv("data" + folder + "/" +data_set + "/carbon_intensity.csv")
        df_pricing = pd.read_csv("data" + folder + "/" +data_set + "/pricing.csv")
        buildingFilenamesList = glob.glob("data/" + folder + "/" +data_set + "/UCam_Building_*"+ ".csv")
        df_Bs = []
        for file in buildingFilenamesList:
            df_B = pd.read_csv(file)
            df_Bs.append(df_B)
        max_incr = max(self.increments)
        
        if (self.num_buildings!=len(buildingFilenamesList)):
            print("Error with initialising cummulative values, inconsistent building number files")
        
        temp_builds = np.zeros((max_incr,self.num_buildings))
        temp_solar = np.zeros((max_incr,self.num_buildings))
        for b_i, df_B in enumerate(df_Bs):
            temp_builds[:,b_i] = df_Bs[b_i].iloc[-max_incr:,7].values
            temp_solar[:,b_i] = df_Bs[b_i].iloc[-max_incr:,11].values
            
        self.cum_arrays = {'loads': temp_builds, 'pv_gens': temp_solar, 'price': df_pricing.iloc[-max_incr:,0].values, 'carbon': df_carbon.iloc[-max_incr:].values}

    def set_cummulative_values(self):
        temp_builds = np.zeros((len(self.increments),self.num_buildings))
        temp_solar = np.zeros((len(self.increments),self.num_buildings))
        temp_carb = np.zeros((len(self.increments),))
        temp_elec = np.zeros((len(self.increments),))
        
        for i, incr in enumerate(self.increments):
            temp_carb[i] = self.cum_arrays.get('carbon')[-incr:].sum()
            temp_elec[i] = self.cum_arrays.get('price')[-incr:].sum()
            for b in range(self.num_buildings):
                temp_builds[i,b] = self.cum_arrays.get('loads')[-incr:,b].sum()
                temp_solar[i,b] = self.cum_arrays.get('pv_gens')[-incr:,b].sum()
        self.cum_values = {'loads': temp_builds, 'pv_gens': temp_solar, 'price': temp_elec, 'carbon': temp_carb}


    def update_cummulative_arrays(self, new_value_c,new_value_e,new_values_b, new_value_s):
        "new values consist of observation values for 19,20,21,24"
        
        temp_carbon = self.cum_arrays.get('carbon')
        temp_carbon = np.roll(temp_carbon,-1)
        temp_carbon [-1] = new_value_c
        
        temp_elec = self.cum_arrays.get('price')
        temp_elec = np.roll(temp_elec,-1)
        temp_elec [-1] = new_value_e
        
        temp_builds = self.cum_arrays.get('loads')
        for b_i,new_b in enumerate(new_values_b):
            temp_builds[:,b_i] = np.roll(temp_builds[:,b_i],-1)
            temp_builds [-1,b_i] = new_b
        temp_solar = self.cum_arrays.get('pv_gens')
        # Artifact - solar unique value, but we use same structure as buildings for now
        for b_i,new_b in enumerate(new_values_b):
            temp_solar[:,b_i] = np.roll(temp_solar[:,b_i],-1)
            temp_solar [-1,b_i] = new_value_s
                
        self.cum_arrays = {'loads': temp_builds, 'pv_gens': temp_solar, 'price': temp_elec, 'carbon': temp_carbon}


    def compute_forecast(self, observations):
        """Compute forecasts given current observation.

        Args:
            observation (List[List]): observation data for current time instance, as
                specified in CityLearn documentation.
                The observation is a list of observations for each building (sub-list),
                where the sub-lists contain values as specified in the ReadMe.md

        Returns:
            predicted_loads (np.array): predicted electrical loads of buildings in each
                period of the planning horizon (kWh) - shape (N,tau)
            predicted_pv_gens (np.array): predicted energy generations of pv panels in each
                period of the planning horizon (kWh) - shape (N,tau)
            predicted_pricing (np.array): predicted grid electricity price in each period
                of the planning horizon ($/kWh) - shape (tau)
            predicted_carbon (np.array): predicted grid electricity carbon intensity in each
                period of the planning horizon (kgCO2/kWh) - shape (tau)
        """

        # ====================================================================
        # insert your forecasting code here
        # ====================================================================
        
        common_features = np.array(observations)[0,[0,1,2,3,7,11,15,19,21,24]]
        building_features = np.array(observations)[:,20]
        # print(np.array(observations))
        predicted_pv_gens = []
        predicted_loads = []
        
        if self.model_version == "_H1":
            # Update rolling arrays of past values and cummulative values
            self.update_cummulative_arrays(common_features[7],common_features[9],building_features, common_features[8])      
            self.set_cummulative_values()
            
            building_cum_vals = self.cum_values.get('loads')
            solar_cum_vals = self.cum_values.get('pv_gens')
            # print(solar_cum_vals)
            # print(solar_cum_vals[:,0])
            
            for build_ind, building_load in enumerate(building_features):

                # Building specific    
                common_features_solar = np.concatenate((common_features,[building_load],solar_cum_vals[:,build_ind])).reshape(1,len(common_features)+1+len(solar_cum_vals[:,build_ind]))
                predicted_pv_gens.append(self.pre_trained_models_solar.predict(common_features_solar))
                common_features_builds = np.concatenate((common_features,[building_load],building_cum_vals[:,build_ind])).reshape(1,len(common_features)+1+len(building_cum_vals[:,build_ind]))
                predicted_loads.append(self.pre_trained_models_Bs[build_ind].predict(common_features_builds))
                
                #Non building specific
                if (build_ind==0):
                    common_features_pricing = np.concatenate((common_features,[building_load],self.cum_values.get('price'))).reshape(1,len(common_features)+1+len(self.cum_values.get('price')))
                    predicted_pricing = np.array(self.pre_trained_models_elec.predict(common_features_pricing)).reshape(self.tau)

                    common_features_carbon = np.concatenate((common_features,[building_load],self.cum_values.get('carbon'))).reshape(1,len(common_features)+1+len(self.cum_values.get('carbon')))
                    predicted_carbon = np.array(self.pre_trained_models_carbon.predict(common_features_carbon)).reshape(self.tau)
                    
            predicted_pv_gens = np.array(predicted_pv_gens).reshape(self.num_buildings,self.tau)
            # predicted_pv_gens = np.array(predicted_pv_gens).reshape(5,12)
            predicted_loads = np.array(predicted_loads).reshape(self.num_buildings,self.tau)
            
        elif self.model_version == "_H2":
            features_solar = [common_features[8]] 
            # features_B = building_features
            features_c = [common_features[7]]
            features_cost = [common_features[9]]
            
            # Update rolling arrays of past values and cummulative values
            self.update_cummulative_arrays(common_features[7],common_features[9],building_features, common_features[8])      
            self.set_cummulative_values()
            building_cum_vals = self.cum_values.get('loads')
            solar_cum_vals = self.cum_values.get('pv_gens')
            
            for build_ind, building_load in enumerate(building_features):
                
                # Building specific    
                common_features_solar = np.concatenate((features_solar,solar_cum_vals[:,build_ind])).reshape(1,len(features_solar)+len(solar_cum_vals[:,build_ind]))
                predicted_pv_gens.append(self.pre_trained_models_solar.predict(common_features_solar))
                common_features_builds = np.concatenate(([building_load],building_cum_vals[:,build_ind])).reshape(1,1+len(building_cum_vals[:,build_ind]))
                predicted_loads.append(self.pre_trained_models_Bs[build_ind].predict(common_features_builds))
                
                #Non building specific
                if (build_ind==0):
                    common_features_pricing = np.concatenate((features_cost,self.cum_values.get('price'))).reshape(1,len(features_cost)+len(self.cum_values.get('price')))
                    predicted_pricing = np.array(self.pre_trained_models_elec.predict(common_features_pricing)).reshape(self.tau)

                    common_features_carbon = np.concatenate((features_c,self.cum_values.get('carbon'))).reshape(1,len(features_c)+len(self.cum_values.get('carbon')))
                    predicted_carbon = np.array(self.pre_trained_models_carbon.predict(common_features_carbon)).reshape(self.tau)
                    
            predicted_pv_gens = np.array(predicted_pv_gens).reshape(self.num_buildings,self.tau)
            # predicted_pv_gens = np.array(predicted_pv_gens).reshape(5,12)
            predicted_loads = np.array(predicted_loads).reshape(self.num_buildings,self.tau)
            
            
        elif self.model_version == "_H3":
            features_solar = common_features
            # features_B = building_features
            features_c = [common_features[7]]
            features_cost = [common_features[9]]
            
            # Update rolling arrays of past values and cummulative values
            self.update_cummulative_arrays(common_features[7],common_features[9],building_features, common_features[8])      
            self.set_cummulative_values()
            building_cum_vals = self.cum_values.get('loads')
            solar_cum_vals = self.cum_values.get('pv_gens')
            
            for build_ind, building_load in enumerate(building_features):
                
                # Building specific    
                # common_features_solar = np.concatenate((features_solar,[building_load],solar_cum_vals[:,build_ind])).reshape(1,len(features_solar)+len(solar_cum_vals[:,build_ind]))
                common_features_solar = np.concatenate((common_features,[building_load],solar_cum_vals[:,build_ind])).reshape(1,len(common_features)+1+len(solar_cum_vals[:,build_ind]))

                predicted_pv_gens.append(self.pre_trained_models_solar.predict(common_features_solar))
                common_features_builds = np.concatenate(([building_load],building_cum_vals[:,build_ind])).reshape(1,1+len(building_cum_vals[:,build_ind]))
                predicted_loads.append(self.pre_trained_models_Bs[build_ind].predict(common_features_builds))
                
                #Non building specific
                if (build_ind==0):
                    common_features_pricing = np.concatenate((features_cost,self.cum_values.get('price'))).reshape(1,len(features_cost)+len(self.cum_values.get('price')))
                    predicted_pricing = np.array(self.pre_trained_models_elec.predict(common_features_pricing)).reshape(self.tau)

                    common_features_carbon = np.concatenate((features_c,self.cum_values.get('carbon'))).reshape(1,len(features_c)+len(self.cum_values.get('carbon')))
                    predicted_carbon = np.array(self.pre_trained_models_carbon.predict(common_features_carbon)).reshape(self.tau)
                    
            predicted_pv_gens = np.array(predicted_pv_gens).reshape(self.num_buildings,self.tau)
            # predicted_pv_gens = np.array(predicted_pv_gens).reshape(5,12)
            predicted_loads = np.array(predicted_loads).reshape(self.num_buildings,self.tau)
            
            
            
                 
            
        else:
                
                            

            for build_ind, building_load in enumerate(building_features):
                common_features_temp = np.append(common_features,[building_load]).reshape(1,len(common_features)+1)
                predicted_pv_gens.append(self.pre_trained_models_solar.predict(common_features_temp))
                # predicted_pv_gens.append(pre_trained_models_solar.predict(common_features_temp))
                predicted_loads.append(self.pre_trained_models_Bs[build_ind].predict(common_features_temp))
                if (build_ind==0):
                    predicted_pricing = np.array(self.pre_trained_models_elec.predict(common_features_temp)).reshape(self.tau)
                    predicted_carbon = np.array(self.pre_trained_models_carbon.predict(common_features_temp)).reshape(self.tau)
                    
            predicted_pv_gens = np.array(predicted_pv_gens).reshape(self.num_buildings,self.tau)
            # predicted_pv_gens = np.array(predicted_pv_gens).reshape(5,12)
            predicted_loads = np.array(predicted_loads).reshape(self.num_buildings,self.tau)
        
            predicted_pv_gens = []
            predicted_loads = []
            
                    
        

        # # dummy forecaster for illustration - delete for your implementation
        # # ====================================================================
        # current_vals = {
        #     'loads': np.array(observations)[:,20],
        #     'pv_gens': np.array(observations)[:,21],
        #     'pricing': np.array(observations)[0,24],
        #     'carbon': np.array(observations)[0,19]
        # }


        # if self.prev_vals['carbon'] is None:
        #     predicted_loads = np.repeat(current_vals['loads'].reshape(self.num_buildings,1),self.tau,axis=1)
        #     # predicted_pv_gens = np.repeat(current_vals['pv_gens'].reshape(self.num_buildings,1),self.tau,axis=1)
        #     predicted_pricing = np.repeat(current_vals['pricing'],self.tau)
        #     predicted_carbon = np.repeat(current_vals['carbon'],self.tau)

        # else:
        #     predict_inds = [t+1 for t in range(self.tau)]

        #     # note, pricing & carbon predictions of all zeros can lead to issues, so clip to 0.01
        #     load_lines = [np.poly1d(np.polyfit([-1,0],[self.prev_vals['loads'][b],current_vals['loads'][b]],deg=1)) for b in range(self.num_buildings)]
        #     predicted_loads = np.array([line(predict_inds) for line in load_lines]).clip(0.01)

        #     pv_gen_lines = [np.poly1d(np.polyfit([-1,0],[self.prev_vals['pv_gens'][b],current_vals['pv_gens'][b]],deg=1)) for b in range(self.num_buildings)]
        #     # predicted_pv_gens = np.array([line(predict_inds) for line in pv_gen_lines]).clip(0)

        #     predicted_pricing = np.poly1d(np.polyfit([-1,0],[self.prev_vals['pricing'],current_vals['pricing']],deg=1))(predict_inds).clip(0.01)

        #     predicted_carbon = np.poly1d(np.polyfit([-1,0],[self.prev_vals['carbon'],current_vals['carbon']],deg=1))(predict_inds).clip(0.01)


        # self.prev_vals = current_vals
        # ====================================================================


        return predicted_loads, predicted_pv_gens, predicted_pricing, predicted_carbon
