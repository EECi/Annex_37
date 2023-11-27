"""Assess forecasting generalisation performance of models."""

import os
import sys
import warnings

from assess_forecasts import assess, save_results

from models import (
    DMSPredictor,
    TFT_Predictor,
    NHiTS_Predictor,
    DeepAR_Predictor,
    LSTM_Predictor,
    GRU_Predictor,
    GRWN_Predictor
)


if __name__ == "__main__":

    # Run using
    # for ($m = 0; $m -le 4; $m++) {for ($bid = 0; $bid -le 14; $bid++) {python -m experiments.generalisation_performance $m $bid}}
    # ==================================================================================================

    m = int(sys.argv[1])
    b = int(sys.argv[2])

    UCam_ids = [0,3,9,11,12,15,16,25,26,32,38,44,45,48,49] # set as list of same int to test model on different buildings

    tau = 48  # model prediction horizon (number of timesteps of data predicted)
    dataset_dir = os.path.join('analysis', 'test')  # dataset directory
    schema_path = os.path.join('data', dataset_dir, 'schema.json')

    results_file = os.path.join('results', 'prediction_tests_generalisation.csv')

    model_names = ['linear','resmlp','conv','TFT','NHiTS']
    expt_names = ['linear','resmlp','conv'] + ['analysis']*2
    predictor_types = [DMSPredictor]*3 + [TFT_Predictor,NHiTS_Predictor]

    model_name = model_names[m]
    expt_name = expt_names[m]
    predictor_type = predictor_types[m]
    train_building_index = UCam_ids[b]

    if predictor_type in [TFT_Predictor,NHiTS_Predictor,DeepAR_Predictor]:
        predictor = predictor_type(model_group_name=expt_name, model_names=[train_building_index]*len(UCam_ids))
    elif predictor_type in [DMSPredictor]:
        predictor = predictor_type(building_indices=UCam_ids, expt_name=os.path.join('analysis',expt_name), load=True)

    print("Assessing forecasts for model %s."%model_name)

    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', module=r'cvxpy')
        results = assess(predictor, schema_path, tau, building_breakdown=True, train_building_index=train_building_index)

    results.update({
        'model_name': model_name,
        'train_building_index': train_building_index,
        'tau': tau
        })

    print(results)

    save_results(results, results_file)