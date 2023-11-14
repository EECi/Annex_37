"""Assess baseline performance of forecasts for model comparison."""

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
    # for ($var = 0; $var -le 5; $var++) {python -m experiments.baseline_forecasting_comparison $var}
    # ==================================================================================================

    index = int(sys.argv[1])

    UCam_ids = [0,3,9,11,12,15,16,25,26,32,38,44,45,48,49] # set as list of same int to test model on different buildings

    tau = 48  # model prediction horizon (number of timesteps of data predicted)
    dataset_dir = os.path.join('analysis', 'test')  # dataset directory
    schema_path = os.path.join('data', dataset_dir, 'schema.json')

    results_file = os.path.join('results', 'prediction_tests_baseline.csv')


    predictor_types = [DMSPredictor]*3 + [TFT_Predictor,NHiTS_Predictor,DeepAR_Predictor]
    model_names = ['linear','resmlp','conv'] + ['analysis']*3

    model_name = model_names[index]
    predictor_type = predictor_types[index]

    if predictor_type in [TFT_Predictor,NHiTS_Predictor,DeepAR_Predictor]:
        predictor = predictor_type(model_group_name=model_name)
    elif predictor_type in [DMSPredictor]:
        predictor = predictor_type(building_indices=UCam_ids, expt_name=os.path.join('analysis',model_name), load=True)

    print("Assessing forecasts for model %s."%model_name)

    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', module=r'cvxpy')
        results = assess(predictor, schema_path, tau, building_breakdown=True, train_building_index=None)

    results.update({
        'model_name': model_name,
        'train_building_index': 'same-train-test',
        'tau': tau
        })

    save_results(results, results_file)