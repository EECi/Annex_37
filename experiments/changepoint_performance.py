"""Assess forecasting performance of changepoint models."""

import os
import sys
import json
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
    test_change_points = []
    test_expt_names = []


    for b_id in UCam_ids:
        expt_name =  os.path.join('analysis','changepoint','linear_b%s-cp-baseline'%b_id)
        test_building_ids.append(b_id)
        test_change_points.append('None')
        test_change_points.append(expt_name)


    with open(os.path.join('data','analysis','changepoint','single_changepoints.json'),'r') as json_file:
        single_changepoints = json.load(json_file)['data']

    for b_id in UCam_ids:
        cp = single_changepoints[str(b_id)]
        if cp is not None:
            expt_name = 'linear_b%s-scp'%b_id
            test_building_ids.append(b_id)
            test_change_points.append(cp)
            test_expt_names.append(expt_name)


    with open(os.path.join('data','analysis','changepoint','multiple_changepoints.json'),'r') as json_file:
        multiple_changepoints = json.load(json_file)['data']

    for b_id in multiple_changepoints.keys():
        b_id = int(b_id)
        # NOTE: '%Y-%m-%d' format allows dates to be sorted as strings
        cps = sorted([multiple_changepoints[str(b_id)][i]['date'] for i in range(len(multiple_changepoints[str(b_id)]))])
        for j,cp in enumerate(cps):
            expt_name =  'linear_b%s-mcp%s'%(b_id,j)
            test_building_ids.append(b_id)
            test_change_points.append(cp)
            test_expt_names.append(expt_name)


    # Assess quality of forecasts for changepoint models
    # ==================================================
    for b_id, cp, expt_name in zip(test_building_ids, test_change_points, test_expt_names):
        linear_predictor = DMSPredictor(building_indices=[b_id], expt_name=os.path.join('analysis','changepoint',expt_name), load=True)
        predictor = BuildingForecastsOnlyWrapper(linear_predictor, tau)

        # adjust schema so only building with changepoint active
        schema_dict = json.load(schema_path)
        for id in UCam_ids:
            if id != b_id:
                schema_dict['buildings']['UCam_Building_%s'%id]['include'] = False

        print("Assessing forecasts for model %s."%expt_name)

        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', module=r'cvxpy')
            results = assess(predictor, schema_dict, tau, building_breakdown=True, train_building_index=None)

        # adjust results dict to place 'None' entry for all untested variables
        metric_names = ['gmnMAE', 'gmnRMSE']

        results['Load Forecasts'] = {
            'UCam_Building_%s'%id: {
                mname: results['Load Forecasts']['UCam_Building_%s'%b_id][mname] if id == b_id else 'None'\
                    for mname in metric_names
            } for id in UCam_ids
        }
        for vname in ['Solar Potential Forecasts','Pricing Forecasts','Carbon Intensity Forecasts']:
            for mname in metric_names:
                results[vname][mname] = 'None'

        results.update({
            'building_id': b_id,
            'change_point': cp,
            'model_name': expt_name+'-cp_%s'%cp,
            'train_building_index': 'same-train-test',
            'tau': tau
            })

        print(results)

        save_results(results, results_file)