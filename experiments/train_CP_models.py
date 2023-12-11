"""Train Linear models on curtailed training datasets for changepoint analysis.

Note: for changepoint analysis 1 year of validation data is used to provide a
greater allowable range for changepoint locations while maintaining sufficient
train & validate dataset sizes."""

import os
import sys
import csv
import json
import time
import glob
import shutil
import pandas as pd
import numpy as np
import torch
import random

from models.dms.predictor import Predictor
from experiments.train_DMS_models import get_mparams


def create_cp_training_datasets(building_id: int, changepoint_timestamp: str):
    """Create temporary train and validate datasets for individual changepoint
    model training. Training dataset start at specified changepoint and data is
    provided for only single building as specified by `building_id`.

    Args:
        building_id (int): Index of building to train model for.
        changepoint_timestamp (str): Timestamp of changepoint to use for creating
        training data. Must be in format '%Y-%m-%d'.
    """

    changepoint_timestamp += ' 00:00:00' # add hour info to timestamp

    # set directory paths
    dir_path = os.path.join('data','analysis','changepoint')
    train_path = os.path.join(dir_path,'v1y','train')
    validate_path = os.path.join(dir_path,'v1y','validate')
    save_path = os.path.join(dir_path,'temp')

    # create output directories
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for ds in ['train','validate']:
        if not os.path.exists(os.path.join(save_path,ds)):
            os.makedirs(os.path.join(save_path,ds))


    # read data
    timestamp_df = pd.read_csv(os.path.join(train_path,'timestamps.csv'))
    building_df = pd.read_csv(os.path.join(train_path,'UCam_Building_%s.csv'%building_id))
    pricing_df = pd.read_csv(os.path.join(train_path,'pricing.csv'))
    carbon_df = pd.read_csv(os.path.join(train_path,'carbon_intensity.csv'))

    # get changepoint index
    cp_id, = timestamp_df[timestamp_df['Timestamp (UTC)'] == changepoint_timestamp].index

    # cut down training files with specified changepoint and
    # save to temporary training dataset directory
    files = ['timestamps.csv','UCam_Building_%s.csv'%building_id,'pricing.csv','carbon_intensity.csv']
    dfs = [timestamp_df,building_df,pricing_df,carbon_df]

    for file,df in zip(files,dfs):
        df.iloc[cp_id:].to_csv(os.path.join(save_path,'train',file),index=False)
    
    # copy validate dataset files for selected building into temporary directory
    for file in files:
        shutil.copy(os.path.join(validate_path,file),os.path.join(save_path,'validate',file))


def create_and_train_cp_model(b_id: int, cp: str, expt_name: str, L: int, T: int):
    """Create Linear model and training data for specified changepoint
    and building, and train.

    Args:
        building_id (int): Index of building to train model for.
        changepoint_timestamp (str): Timestamp of changepoint to use for creating
        training data. Must be in format '%Y-%m-%d'.
        expt_name (str): Path of dir to save model training logs.
        L (int): The length of the input sequence for linear model.
        T (int): Length of planning horizon for linear model.
    """

    print("Training changepoint Linear model for building %s with changepoint %s\n"%(b_id,cp))

    dataset_dir = os.path.join('data','analysis','changepoint','temp')

    create_cp_training_datasets(b_id, cp)

    start = time.time()
    predictor = Predictor(get_mparams('linear',L,T), [b_id], L, T, expt_name, load=False)
    for var in ['solar', 'carbon', 'price']:
        predictor.training_order.remove(var)
    predictor.train(patience=100, max_epoch=500, dataset_dir=dataset_dir)
    end = time.time()

    print("Train time:", end-start)

    with open(os.path.join('models','dms','resources',expt_name,'training_time.csv'),'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Create & train time (s)', end-start])

    # clear learning rate checkpoints from current directory
    lr_checkpoint_list = glob.glob('.lr_find*')
    for lr_checkpoint in lr_checkpoint_list:
        os.remove(lr_checkpoint)



if __name__ == '__main__':

    # Run using
    # for ($var = 0; $var -le 2; $var++) {python -m experiments.train_CP_models $var}
    # or split into separate runs
    # ==================================================================================================

    index = int(sys.argv[1])

    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    L = 168
    T = 48

    UCam_ids = [0,3,9,11,12,15,16,25,26,32,38,44,45,48,49] # set as list of same int to test model on different buildings

    if index == 0:
        # Train baselime models for changepoint analysis
        # i.e. models trained using 7yr train set and 1yr test set
        # ========================================================
        for b_id in UCam_ids:
            expt_name =  os.path.join('analysis','changepoint','linear_b%s-cp-baseline'%b_id)
            create_and_train_cp_model(b_id, '2010-01-01', expt_name, L, T)

    elif index == 1:
        # Train models for single changepoint analysis
        # ============================================
        with open(os.path.join('data','analysis','changepoint','single_changepoints.json'),'r') as json_file:
            single_changepoints = json.load(json_file)['data']

        for b_id in UCam_ids:
            cp = single_changepoints[str(b_id)]
            if cp is not None:
                expt_name =  os.path.join('analysis','changepoint','linear_b%s-scp'%b_id)
                create_and_train_cp_model(b_id, cp, expt_name, L ,T)

    elif index == 2:
        # Train models for multiple changepoint analysis
        # ==============================================
        with open(os.path.join('data','analysis','changepoint','multiple_changepoints.json'),'r') as json_file:
            multiple_changepoints = json.load(json_file)['data']

        for b_id in multiple_changepoints.keys():
            b_id = int(b_id)
            # NOTE: '%Y-%m-%d' format allows dates to be sorted as strings
            cps = sorted([multiple_changepoints[str(b_id)][i]['date'] for i in range(len(multiple_changepoints[str(b_id)]))])
            for j,cp in enumerate(cps):
                expt_name =  os.path.join('analysis','changepoint','linear_b%s-mcp%s'%(b_id,j))
                create_and_train_cp_model(b_id, cp, expt_name, L, T)

    else:
        raise ValueError("Command line index argument must be in range 0-2.")