"""Train pytorch-forecast models required to run experiments."""

import os
import json

from models import TFT_Predictor, NHiTS_Predictor, DeepAR_Predictor
from models.TFModels.create_train_TF_group import main as train_TF_models


if __name__ == '__main__':

    # Train baseline models
    # ==================================================================================================================
    # set paths for train & validation datasets for training
    dataset_path = os.path.join('data','analysis')
    train_path = os.path.join(dataset_path,'train')
    val_path = os.path.join(dataset_path,'validate')

    # grab building ids in specified dataset
    with open(os.path.join(train_path,'metadata_ext.json')) as json_file:
        UCam_ids = json.load(json_file)["UCam_building_ids"]

    # set model group parameters
    model_group_name = 'analysis'
    model_architectures = ['TFT','NHiTS','DeepAR']
    predictor_models = [TFT_Predictor, NHiTS_Predictor, DeepAR_Predictor]

    for arch, model in zip(model_architectures, predictor_models):
        train_TF_models(model_group_name, arch, model, UCam_ids, train_path, val_path)


    # Train reduced data models
    # ==================================================================================================================
    model_architecture = 'TFT'
    predictor_model = TFT_Predictor

    dataset_path = os.path.join('data','analysis','reduced')
    data_lengths = ['rd4y','rd2y']

    for rd in data_lengths:
        train_path = os.path.join(dataset_path,rd,'train')
        val_path = os.path.join(dataset_path,rd,'validate')
        model_group_name = 'analysis-%s'%rd
        train_TF_models(model_group_name, model_architecture, predictor_model, UCam_ids, train_path, val_path)