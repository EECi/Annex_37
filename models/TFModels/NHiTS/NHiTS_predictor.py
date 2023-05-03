"""Implementation of N-HiTS predictor model class."""

from ..TFPredictor import TF_Predictor
from pytorch_forecasting import NHiTS, QuantileLoss



class NHiTS_Predictor(TF_Predictor):
    """Implementation of N-HiTS-based prediction model for the CityLearn LinMPC controller."""

    def __init__(self, *args, **kwargs) -> None:

        self.model_architecture = 'NHiTS'
        self.pytorch_forecasting_model_class = NHiTS
        self.model_constructor_kwargs = {
            # architecture hyperparameters
            'hidden_size': 128,
            'weight_decay': 1e-2,
            'backcast_loss_ratio': 0.0,
            'dropout': 0.1,
            # loss metric to optimize
            'loss': QuantileLoss(),
            # set optimizer
            'optimizer': 'AdamW',
            # optimizer parameters
            'reduce_on_plateau_patience': 3,
        }

        super().__init__(*args, **kwargs)


    def _specify_time_varying_unknown_reals(self, model_type: str):
        """Specify required time varying unknown reals for model."""

        if model_type == 'load':
            time_varying_unknown_reals = [self.load_col_name]
        elif model_type == 'solar':
            time_varying_unknown_reals = [self.solar_col_name]
        elif model_type == 'pricing':
            time_varying_unknown_reals = [self.pricing_col_name]
        elif model_type == 'carbon':
            time_varying_unknown_reals = [self.carbon_col_name]

        return time_varying_unknown_reals
