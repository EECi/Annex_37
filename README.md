This package is based on Pat’s `DMS` framework and focuses on linear models. The current code is specifically tuned for linear models.

# Feature Number Explanation

In the `Feature` and `Feature_N` folders, we examine the impacts of selected feature numbers on model performance. The `Feature` folder represents feature parameters without normalization, whereas `Feature_N` includes feature parameters after normalization.
## Understanding Feature Selection for Predictive Variables
Figure 1 displays the pairwise correlation of input features for building_0.

**Fig.1**: Correlation Matrix for Building 0
![Correlation Matrix for Building 0](/plots/correlation.png)

Figure 2 shows the most significant influencing features for the predictive variables (load, solar, price, and carbon). The significant influencing features vary for different predictive variables. For instance, when the feature number is 3, the chosen features for 'load' include load, DifSolar, and for 'price', the features include price, load, and Hour. For 'solar', the features include solar, DirSolar, and DifSolar. For 'carbon', the features include carbon, Temp, and Price. These selected features are used as inputs for model training.

**Fig.2**: Significant Factors Affecting Load, Solar, Price, and Carbon
![Significant Factors](/plots/significantfactors.png)

 
<br>

# Online Learning
In the `Online` folder, we investigate the impacts of online learning on improving model forecasting. A buffer is created to store the data of the past 1 year, and the model is retrained at given updating frequencies.

<br>

The `plots` folder includes all the results for the impacts of feature number and online learning frequency. You are able to recreate the figures using the `results_plot.py`.
