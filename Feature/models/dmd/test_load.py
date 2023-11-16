"""Test loading of DMD predictor implementation."""

import os
from models import DMDPredictor

if __name__ == '__main__':

    test_data = os.path.join('data', 'example')
    predictor = DMDPredictor(dataset_dir=test_data)