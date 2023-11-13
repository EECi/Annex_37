"""Assess forecasting performance of changepoint models."""

import os
import sys
import warnings

import numpy as np
from assess_forecasts import assess, save_results

from models import DMSPredictor



def BuildingForecastsOnlyWrapper():
    """Prediction model wrapper that gets building forecasts from a model
    and returns alongside zeros for other variables when forecasts are computed.
    """
    def __init__(self, predictor, tau: int):

        self.predictor = predictor
        self.tau = tau

    def compute_forecast(self, observations, train_building_index=None):

        predicted_loads,_,_,_ = self.predictor.compute_forecast(observations, train_building_index)

        return predicted_loads, np.zeros((self.tau)), np.zeros((self.tau)), np.zeros((self.tau))



if __name__ == "__main__":

    UCam_ids = [0,3,9,11,12,15,16,25,26,32,38,44,45,48,49] # set as list of same int to test model on different buildings

    tau = 48  # model prediction horizon (number of timesteps of data predicted)
    dataset_dir = os.path.join('analysis', 'test')  # dataset directory
    schema_path = os.path.join('data', dataset_dir, 'schema.json')

    results_file = os.path.join('results', 'prediction_tests_changepoint.csv')


    # Construct schedule of changepoint models to test
    # ===========================================================
    test_building_ids = []
    test_expt_names = []

    single_changepoints = [...] # get from Monika spreadsheet
    for b_id, cp in zip(UCam_ids,single_changepoints):
        expt_name = 'linear_b%s-scp'%b_id
        test_building_ids.append(b_id)
        test_expt_names.append(expt_name)

    multiple_changepoints = {
        ...: ... # b_id: list of changepoints
    } # get from Monika spreadsheet
    for b_id in multiple_changepoints.keys():
        for j,cp in enumerate(multiple_changepoints[b_id]):
            expt_name =  'linear_b%s-mcp%s'%(b_id,j)
            test_building_ids.append(b_id)
            test_expt_names.append(expt_name)


    # Assess quality of forecasts for changepoint models
    # ==================================================
    for b_id, expt_name in zip(test_building_ids, test_expt_names):
        linear_predictor = DMSPredictor(building_indices=[b_id], expt_name=os.path.join('analysis','changepoint',expt_name), load=True)
        predictor = BuildingForecastsOnlyWrapper(linear_predictor, tau)

        print("Assessing forecasts for model %s."%expt_name)

        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', module=r'cvxpy')
            results = assess(predictor, schema_path, tau, building_breakdown=True, train_building_index=None)

        results.update({
            'model_name': expt_name,
            'train_building_index': 'same-train-test',
            'tau': tau
            })

        print(results)

        save_results(results, results_file)