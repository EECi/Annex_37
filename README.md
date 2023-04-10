# What's in this branch?

...

For information on the the setup of the task, see the `README` on the `main` branch.

# EECi's approach

... what's everyone doing? ...

| Name    | Research Question                                                                                                                                                                                                                                         | Model/Method                                                                                                                                                                     | Directory   | Progress        |
| :------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ----------- | ------------------------------------------------------------ |
| Example | Can we build the infrastructure?                                                                                                                                                                                                                          | Linear Interpolation                                                                                                                                                             | [`example`](models/example/README.md) | Complete ✅     |
| Max     | How does the volume of available of training data, and<br />the similarity of training data to test data, affect prediciton<br />performance for neural methods? (How much data do we<br />need for training, and can we use data from other buildings?) | - [Temporal Fusion Transformers](https://arxiv.org/abs/1912.09363) <br> - [Neural Hierarchical Interpolation for Time Series Forecasting (N-HiTS)](https://arxiv.org/abs/2201.12886) | tbc         | Getting started |
| Pat     |- Which methods are best for time-series forecasting?<br> - Why is this the case (inductive biases)? | - Direct multi-step prediction<br> - Confidence score prediction <br> - Ensembling <br> - Action-supervised forecasting| [`dms`](models/dms/README.md)         | - Direct Multi-step forecasting with MLP implemented ✅ <br> - About to start confidence score stuff 👨‍🔧|
| Nick    |- Can we create generalised building-type training data sets to use for “typical buildings” in absence of building-specific training data?<br> - How do we fine-tune a pre-trained predictor? <br>How building-specific does the training set need to be?<br> - Comparison of predictors trained with specific building data vs generalised building data | - Any supervised model, especially easy/fast to train<br> - Statistical comparison methods & metrics| tbc         ||
| Monika  |- How do we classify the behaviours/trends within the time-series?<br> - How do we detect change in behaviours/trends in the time-series?<br> - How does model selection change upon changes in trend? |  Online change point detection algorithms, e.g. Bayesian Online Changepoint Detection (leading to Bayesian On-line Changepoint Detection with Model Selection (BOCPDMS) if time permits) | tbc         |                 |
| Rebecca |- What are the similarities and differences between the demand profiles for different buildings? | FDA | tbc         |                 |
| Zack    |-Can we infer representative dynamics from observed data to build more robust and interpretable predictors? How does this influence training volume efficiency?<br> - Can we leverage coarse Physics-based models to improve training efficiency and predictor robustness? <br> -  Can we obtain simplified models with the least number of tunable hyperparameters possible as required in control applications? | - Dynamic Mode Decomposition with control (DMDc) <br> - High-order Dynamic Mode Decomposition (HODMD) <br> -  LSTM/PINN with RC models | tbc         |                 |
| Chaoqun |                                                                                                                                                                                                                                                           |                                                                                                                                                                                  | tbc         |                 |

# How to use the branch

... [`leaderboard`](outputs/leaderboard.md)

... explain structure of branch directories ...

... explain how to add a model ...
