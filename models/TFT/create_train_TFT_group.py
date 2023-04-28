"""Create and train TFT model group."""

import os
import json
import warnings
from models.TFT.TFT_predictor import TFT_Predictor


def main(model_group_name, UCam_ids, train_path, val_path):

    # safety check the model creation
    model_group_path = os.path.join('models','TFT','resources','model_logs',model_group_name)
    if os.path.exists(model_group_path):
        warnings.warn("Warning: Log directories already exist for the model group `{}`. By continuing you will overwrite this model.".format(TFT_group.model_group_path))
        if input("Are you sure you want to overwrite this model? [y/n]") not in ['y','yes','Y','Yes','YES']:
            print("Aborting model creation.")
            return

    # initialise new model group
    TFT_group = TFT_Predictor(model_group_name=model_group_name,load=False)

    # construct list of required models (use default name formatting)
    model_types = []
    model_names = []
    building_indices = []

    for i,id in enumerate(UCam_ids):
        # load models
        model_types.append('load')
        model_names.append(f'load_{id}')
        building_indices.append(i)
        # solar models
        model_types.append('solar')
        model_names.append(f'solar_{id}')
        building_indices.append(i)
    # pricing & carbon models
    for m in ['pricing','carbon']:
        model_types.append(m)
        model_names.append(m)
        building_indices.append(None)

    # create and train models
    for model_type, model_name, index in zip(model_types, model_names, building_indices):
        train_ds,val_ds = TFT_group.format_CityLearn_datasets([train_path,val_path], model_type=model_type, building_index=index)
        tft_model = TFT_group.new_model(model_name,model_type,train_ds,pre_confirm=True)
        TFT_group.train_model(model_name,model_type,train_ds,val_ds)



if __name__ == '__main__':

    # set paths for train & validation datasets for training
    dataset_path = os.path.join('data','example')
    train_path = os.path.join(dataset_path,'train')
    val_path = os.path.join(dataset_path,'validate')

    # grab building ids in specified dataset
    with open(os.path.join(train_path,'metadata_ext.json')) as json_file:
        UCam_ids = json.load(json_file)["UCam_building_ids"]

    # set TFT model group parameters
    model_group_name = 'test'

    main(model_group_name, UCam_ids, train_path, val_path)