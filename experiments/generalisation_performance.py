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
    # for ($m = 0; $m -le 5; $m++) {for ($bid = 0; $bid -le 14; $bid++) {python -m experiments.generalisation_performance $m $bid}}
    # ==================================================================================================

    m = int(sys.argv[1])
    b_id = int(sys.argv[2])

    UCam_ids = [0,3,9,11,12,15,16,25,26,32,38,44,45,48,49] # set as list of same int to test model on different buildings

    tau = 48  # model prediction horizon (number of timesteps of data predicted)
    dataset_dir = os.path.join('analysis', 'test')  # dataset directory
    schema_path = os.path.join('data', dataset_dir, 'schema.json')

    results_file = os.path.join('results', 'test.csv')

    predictor_types = [DMSPredictor]*3 + [TFT_Predictor,NHiTS_Predictor,DeepAR_Predictor]
    model_names = ['linear','resmlp','conv'] + ['baseline']*3

    predictor_type = predictor_types[m]
    model_name = os.path.join('analysis',model_names[m])
    train_building_index = b_id

    if predictor_type in [TFT_Predictor,NHiTS_Predictor,DeepAR_Predictor]:
        predictor = predictor_type(model_group_name=model_name, model_names=[b_id]*len(UCam_ids))
    elif predictor_type in [DMSPredictor]:
        predictor = predictor_type(building_indices=UCam_ids, expt_name=model_name, load=True)

    print("Assessing forecasts for model %s."%model_name)

    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', module=r'cvxpy')
        results = assess(predictor, schema_path, tau, building_breakdown=True, train_building_index=train_building_index)

    results.update({
        'model_name': model_name,
        'train_building_index': 'same-train-test',
        'tau': tau
        })

    print(results)

    save_results(results, results_file)