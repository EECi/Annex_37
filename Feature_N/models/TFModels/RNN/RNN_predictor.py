"""Implementation of RNN predictor model class."""

from ..TFPredictor import TF_Predictor
from pytorch_forecasting import RecurrentNetwork, RMSE



class RNN_Predictor(TF_Predictor):
    """Implementation of RNN-based prediction model for the CityLearn LinMPC controller."""

    def __init__(self, *args, **kwargs) -> None:

        if not hasattr(self, 'cell_type'): raise NotImplementedError # must be defined in child model class

        self.model_architecture = 'RNN'
        self.pytorch_forecasting_model_class = RecurrentNetwork
        self.model_constructor_kwargs = {
            # architecture hyperparameters
            'cell_type': self.cell_type,
            'hidden_size': 64,
            'rnn_layers': 3,
            'dropout': 0.1,
            # loss metric to optimize
            'loss': RMSE(quantiles=[0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]),
            # set optimizer
            'optimizer': 'adam',
            # optimizer parameters
            'reduce_on_plateau_patience': 3
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



class LSTM_Predictor(RNN_Predictor):
    """Implementation of LSTM variant of RNN-based prediction model."""

    def __init__(self, *args, **kwargs) -> None:
        self.cell_type = 'LSTM'
        super().__init__(*args, **kwargs)


class GRU_Predictor(RNN_Predictor):
    """Implementation of GRU variant of RNN-based prediction model."""

    def __init__(self, *args, **kwargs) -> None:
        self.cell_type = 'GRU'
        super().__init__(*args, **kwargs)