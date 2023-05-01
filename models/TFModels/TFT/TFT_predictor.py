"""Implementation of TFT predictor model class."""

from ..TFPredictor import TF_Predictor
from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss



class TFT_Predictor(TF_Predictor):
    """Implementation of TFT-based prediction model for the CityLearn LinMPC controller."""

    def __init__(self, **kwargs) -> None:

        self.model_architecture = 'TFT'
        self.pytorch_forecasting_model_class = TemporalFusionTransformer
        self.model_constructor_kwargs = {
            # architecture hyperparameters
            'hidden_size': 48,
            'attention_head_size': 4,
            'dropout': 0.1,
            #if 'hidden_continuous_size' not in kwargs.keys(): kwargs['hidden_continuous_size'] = 16
            # loss metric to optimize
            'loss': QuantileLoss(),
            # set optimizer
            'optimizer': 'adam',
            # optimizer parameters
            'reduce_on_plateau_patience': 3
        }

        super().__init__(**kwargs)


    def _specify_time_varying_unknown_reals(self, model_type: str):
        """Specify required time varying unknown reals for model."""

        if model_type == 'load':
            time_varying_unknown_reals = [self.load_col_name,self.temp_col_name]
        elif model_type == 'solar':
            time_varying_unknown_reals = [self.solar_col_name,self.dif_irad_col_name,self.dir_irad_col_name]
        elif model_type == 'pricing':
            time_varying_unknown_reals = [self.pricing_col_name]
        elif model_type == 'carbon':
            time_varying_unknown_reals = [self.carbon_col_name]

        return time_varying_unknown_reals
