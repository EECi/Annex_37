"""Implementation of abstract parent predictor model class to define required interfaces."""

from abc import ABC, abstractmethod

class BasePredictorModel(ABC):
    """Abstract class defining interfaces required by prediction
    model implementations."""

    @abstractmethod
    def load(self):
        """Method to load pre-trained model from save file."""
        pass

    @abstractmethod
    def compute_forecast(self, observations):
        """Method to perform inference, generating predictions given
        current observations."""
        pass
