"""Train pytorch-forecast models required to run experiments."""

import os
import sys
import json

from models import TFT_Predictor, NHiTS_Predictor, DeepAR_Predictor
from models.TFModels.create_train_TF_group import main as train_TF_models


if __name__ == '__main__':

    # Run using
    # for ($m = 0; $m -le 2; $m++) {for ($rd = 0; $rd -le 2; $rd++) {python -m experiments.train_TF_models $m $rd}}
    # ==================================================================================================

    m = int(sys.argv[1])
    rd = int(sys.argv[2])

    UCam_ids = [0,3,9,11,12,15,16,25,26,32,38,44,45,48,49] # set as list of same int to test model on different buildings

    # set model group parameters
    model_architectures = ['TFT','NHiTS','DeepAR']
    predictor_models = [TFT_Predictor, NHiTS_Predictor, DeepAR_Predictor]
    data_lengths = ['baseline','rd4y','rd2y']

    arch = model_architectures[m]
    model = predictor_models[m]
    dl = data_lengths[rd]

    if dl == 'baseline':
        model_group_name = 'analysis'
        dataset_dir = os.path.join('data','analysis')
    else:
        model_group_name = 'analysis-%s'%dl
        dataset_dir = os.path.join('data','analysis','reduced',dl)
    train_path = os.path.join(dataset_dir,'train')
    val_path = os.path.join(dataset_dir,'validate')

    train_TF_models(model_group_name, arch, model, UCam_ids, train_path, val_path)