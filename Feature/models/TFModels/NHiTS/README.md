# Neural Hierarchical Interpolation for Time Series Forecasting (N-HiTS)

N-HiTS, proposed in [this paper](https://arxiv.org/abs/2201.12886), learns a set of basis functions at different frequencies that describe the underlying patterns in the training data, and produces forecasts by using heirarchical interpolation to combine predictions from the basis functions in a computationally efficient manner. A explanation of the operation of the model is available in [this article](https://towardsdatascience.com/all-about-n-hits-the-latest-breakthrough-in-time-series-forecasting-a8ddcb27b0d5).

An example usage of the model from the pytorch-forecasting docs is available [here](https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/nhits.html).