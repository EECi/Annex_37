import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
##% plot the performance with online updating
# Load the Excel file
def extract_number(model_name):
    # Splitting by '_' and taking the last part (e.g., 'f1', 'f11')
    # Then taking the substring from the 1st character to skip 'f' and convert to int
    return int(model_name.split('_')[-1][:-1])
file_path = 'Online/results/test_online.csv'
df = pd.read_csv(file_path)
df['Sort Key'] = df['Model Name'].apply(extract_number)

# Sort the DataFrame by the new 'Sort Key' column
df = df.sort_values(by='Sort Key')
# Optionally, you can drop the 'Sort Key' column after sorting
df.drop(columns=['Sort Key'], inplace=True)
Matrices=['gmnMAE', 'gmnRMSE']
months=[0,1,3,6,9,12]
def rename_model(model_name):
    for i in months:
        if model_name == f'analysis/linear_1_{i}m':
            return f'{i} mon'
    return model_name  # Returns the original name if no match is found
# Renaming the model names based on the naming convention
df_renamed = df.copy()
df_renamed['Model Name'] = df_renamed['Model Name'].apply(rename_model)
df_renamed = (df_renamed [~df_renamed ['Model Name'].str.contains('linear')])
palette_all = ['#E50068', '#A0006E', '#ffbfeb', '#6E6F73', '#e50900', '#77e500', '#a800e5', '#4d4e51', '#54a000', '#7500a0']
# Redefining the color palette for the renamed model names
updated_palette = {f'{i} mon': color for i, color in zip(months, palette_all)}

# updated_palette will now correctly map each feature number to its color
# Extracting relevant columns for each plot
load_columns = ['L0', 'L3', 'L9', 'L11', 'L12', 'L15', 'L16', 'L25', 'L26', 'L32', 'L38', 'L44', 'L45', 'L48', 'L49']
s_column = ['S']
p_column = ['P']
c_column = ['C']
# Filter out rows where 'Model Name' is 'baseline'
df_baseline = df_renamed[df_renamed['Model Name'] == '0 mon']
df_baseline['Model Name']=df_baseline['Model Name'].replace({'0 mon': 'baseline'})
# Filter out rows where 'Model Name' is not 'baseline'
df_non_baseline = df_renamed[df_renamed['Model Name'] != '0 mon']
# Concatenate the non-baseline DataFrame with the baseline DataFrame
df_renamed= pd.concat([df_non_baseline, df_baseline])
# Replace '0 mon' with 'baseline'
updated_palette['baseline'] = updated_palette.pop('0 mon')

for matrix in Matrices:
    # Filtering the dataframe for the specific metric
    df_filtered = df_renamed[df_renamed['Metric'] == matrix]
    # Preparing data for each plot
    data_load = df_filtered[['Model Name'] + load_columns].melt(id_vars=['Model Name'], var_name='Load',
                                                                value_name='Value')
    data_s = df_filtered[['Model Name'] + s_column].melt(id_vars=['Model Name'], var_name='Solar', value_name='Value')
    data_p = df_filtered[['Model Name'] + p_column].melt(id_vars=['Model Name'], var_name='Price', value_name='Value')
    data_c = df_filtered[['Model Name'] + c_column].melt(id_vars=['Model Name'], var_name='Carbon', value_name='Value')

    # Plotting
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), gridspec_kw={'width_ratios': [4, 1, 1, 1]})

    # Load plot
    sns.barplot(x='Load', y='Value', hue='Model Name', data=data_load, palette=updated_palette, ax=axes[0])
    axes[0].set_title('Load')
    axes[0].tick_params(axis='x', rotation=90)
    axes[0].set_ylabel(matrix)
    axes[0].set_xlabel('Building')

    # Solar plot
    sns.barplot(x='Solar', y='Value', hue='Model Name', data=data_s, palette=updated_palette, ax=axes[1])
    axes[1].set_title('Solar')
    axes[1].get_legend().remove()
    axes[1].tick_params(labelbottom=False)
    axes[1].set_xlabel('')
    axes[1].set_ylabel(matrix)

    # Price plot
    sns.barplot(x='Price', y='Value', hue='Model Name', data=data_p, palette=updated_palette, ax=axes[2])
    axes[2].set_title('Price')
    axes[2].get_legend().remove()
    axes[2].tick_params(labelbottom=False)
    axes[2].set_xlabel('')
    axes[2].set_ylabel(matrix)

    # Carbon plot
    sns.barplot(x='Carbon', y='Value', hue='Model Name', data=data_c, palette=updated_palette, ax=axes[3])
    axes[3].set_title('Carbon')
    axes[3].get_legend().remove()
    axes[3].tick_params(labelbottom=False)
    axes[3].set_xlabel('')
    axes[3].set_ylabel(matrix)

    # Remove the legend from all subplots and place one legend outside
    # for ax in axes:
    #     ax.get_legend().remove()
    # handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.05, 1))

    plt.tight_layout()
    plt.savefig('plots/Updating frequency_'+matrix+'.png' , dpi=300)
    plt.show()
##% plot correlation map
import matplotlib.pyplot as plt
import seaborn as sns
dataset_dir=os.path.join('Feature','data', 'analysis')
version='train'
building_index=0
file_ucam = os.path.join(dataset_dir, version, f'UCam_Building_{building_index}.csv')
file_carbon_intensity = os.path.join(dataset_dir, version, f'carbon_intensity.csv')
file_pricing = os.path.join(dataset_dir, version, f'pricing.csv')
file_weather = os.path.join(dataset_dir, version, f'weather.csv')
# Reading the specific columns from each file
ucam_specific_cols = pd.read_csv(file_ucam,
                                 usecols=['Month', 'Hour', 'Day Type',
                                          'Equipment Electric Power [kWh]', 'Solar Generation [W/kW]', ])
pricing_specific_col = pd.read_csv(file_pricing, usecols=['Electricity Pricing [£/kWh]'])
carbon_intensity_specific_col = pd.read_csv(
    file_carbon_intensity)  # Assuming the relevant column is the only one in this file
weather_specific_cols = pd.read_csv(file_weather, usecols=[0, 1, 2, 3])  # Reading the first 4 columns
# Merging the dataframes based on the row index
merged_specific_cols_df = pd.concat(
    [ucam_specific_cols, pricing_specific_col, carbon_intensity_specific_col, weather_specific_cols],
    axis=1)
merged_specific_cols_df.columns = ['Month', 'Hour', 'Day Type', 'load', 'solar', 'price', 'carbon', 'Tem',
                                   'Hum', 'DifSolar', 'DirSolar']
# New order of columns
new_order = ['Month', 'Hour', 'Day Type', 'Tem', 'Hum', 'DifSolar', 'DirSolar', 'load', 'solar', 'price', 'carbon']

# Reindexing the DataFrame columns according to the new order
merged_specific_cols_df = merged_specific_cols_df[new_order]

correlation_matrix_specific = merged_specific_cols_df.corr()


# Assuming 'correlation_matrix_specific' is your calculated correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_specific, annot=True, fmt='.2f', cmap='coolwarm',
            xticklabels=correlation_matrix_specific.columns, yticklabels=correlation_matrix_specific.columns)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('plots/correlation.png', dpi=300)
plt.show()


##% ranking the influential factors for carbon, load, price and solar
# For the subbarplots for 'load', 'solar', 'price', 'carbon'
variables = ['load', 'solar', 'price', 'carbon']
n_vars = len(variables)

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Flatten axes array for easy iteration
axes = axes.flatten()

# Iterate over each variable and create a subplot for each
for i, var in enumerate(variables):
    # Extract correlation data and sort by absolute value
    corr_series = correlation_matrix_specific[var]  # Keeping self-correlation for now
    corr_series = corr_series.abs().sort_values(ascending=True)  # Sort by absolute value

    # Plot barplot
    sns.barplot(x=corr_series.values, y=corr_series.index, ax=axes[i])
    axes[i].set_title(f'Significant Factors Affecting {var.capitalize()}')

plt.tight_layout()
plt.savefig('plots/significantfactors.png', dpi=300)
plt.show()

##% plot model performance with different feature numbers and normailised inputs
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Define a function to extract the numeric part from the 'Model Name'
def extract_number(model_name):
    # Splitting by '_' and taking the last part (e.g., 'f1', 'f11')
    # Then taking the substring from the 1st character to skip 'f' and convert to int
    return int(model_name.split('_')[-1][1:-1])
# Re-running the plotting code with updated color palette and file path
# Reload the data with the correct file path
df = pd.read_csv('Feature_N/results/test_normalise.csv')
# Apply this function to create a new sorting column
df['Sort Key'] = df['Model Name'].apply(extract_number)


# Sort the DataFrame by the new 'Sort Key' column
df = df.sort_values(by='Sort Key')
# Optionally, you can drop the 'Sort Key' column after sorting
df.drop(columns=['Sort Key'], inplace=True)
# Matrices to be plotted
Matrices = ['gmnMAE', 'gmnRMSE']
feature_numbers=[1,2,5,11]
def rename_model(model_name):
    for i in feature_numbers:
        if model_name == f'analysis/linear_1_f{i}N':
            return f'{i} feature'
    return model_name  # Returns the original name if no match is found
# Renaming the model names based on the naming convention
df_renamed = df.copy()
df_renamed['Model Name'] = df_renamed['Model Name'].apply(rename_model)
df_renamed = (df_renamed [~df_renamed ['Model Name'].str.contains('linear')])
palette_all = ['#E50068', '#A0006E', '#ffbfeb', '#6E6F73', '#e50900', '#77e500', '#a800e5', '#4d4e51', '#54a000', '#7500a0']
# Redefining the color palette for the renamed model names
updated_palette = {f'{i} feature': color for i, color in zip(feature_numbers, palette_all)}
# updated_palette will now correctly map each feature number to its color
# Extracting relevant columns for each plot
load_columns = ['L0', 'L3', 'L9', 'L11', 'L12', 'L15', 'L16', 'L25', 'L26', 'L32', 'L38', 'L44', 'L45', 'L48', 'L49']
s_column = ['S']
p_column = ['P']
c_column = ['C']

for matrix in Matrices:
    # Filtering the dataframe for the specific metric
    df_filtered = df_renamed[df_renamed['Metric'] == matrix]
    # Preparing data for each plot
    data_load = df_filtered[['Model Name'] + load_columns].melt(id_vars=['Model Name'], var_name='Load', value_name='Value')
    data_s = df_filtered[['Model Name'] + s_column].melt(id_vars=['Model Name'], var_name='Solar', value_name='Value')
    data_p = df_filtered[['Model Name'] + p_column].melt(id_vars=['Model Name'], var_name='Price', value_name='Value')
    data_c = df_filtered[['Model Name'] + c_column].melt(id_vars=['Model Name'], var_name='Carbon', value_name='Value')

    # Plotting
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), gridspec_kw={'width_ratios': [4, 1, 1, 1]})

    # Load plot
    sns.barplot(x='Load', y='Value', hue='Model Name', data=data_load, palette=updated_palette, ax=axes[0])
    axes[0].set_title('Normalised Feature - Load')
    axes[0].tick_params(axis='x', rotation=90)
    axes[0].set_ylabel(matrix)
    axes[0].set_xlabel('Building')

    # Solar plot
    sns.barplot(x='Solar', y='Value', hue='Model Name', data=data_s, palette=updated_palette, ax=axes[1])
    axes[1].set_title('Normalised Feature - Solar')
    axes[1].get_legend().remove()
    axes[1].tick_params(labelbottom=False)
    axes[1].set_xlabel('')
    axes[1].set_ylabel(matrix)

    # Price plot
    sns.barplot(x='Price', y='Value', hue='Model Name', data=data_p, palette=updated_palette, ax=axes[2])
    axes[2].set_title('Normalised Feature - Price')
    axes[2].get_legend().remove()
    axes[2].tick_params(labelbottom=False)
    axes[2].set_xlabel('')
    axes[2].set_ylabel(matrix)

    # Carbon plot
    sns.barplot(x='Carbon', y='Value', hue='Model Name', data=data_c, palette=updated_palette, ax=axes[3])
    axes[3].set_title('Normalised Feature - Carbon')
    axes[3].get_legend().remove()
    axes[3].tick_params(labelbottom=False)
    axes[3].set_xlabel('')
    axes[3].set_ylabel(matrix)

    plt.tight_layout()
    plt.savefig('plots/Normalised Feature_' + matrix + '.png', dpi=300)
    plt.show()


##% plot model performance with different feature numbers and without normailised inputs
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Define a function to extract the numeric part from the 'Model Name'
def extract_number(model_name):
    # Splitting by '_' and taking the last part (e.g., 'f1', 'f11')
    # Then taking the substring from the 1st character to skip 'f' and convert to int
    return int(model_name.split('_')[-1][1:])
# Re-running the plotting code with updated color palette and file path
# Reload the data with the correct file path
df = pd.read_csv('Feature/results/test_unnormalised.csv')
# Apply this function to create a new sorting column
df['Sort Key'] = df['Model Name'].apply(extract_number)
# Sort the DataFrame by the new 'Sort Key' column
df = df.sort_values(by='Sort Key')
# Optionally, you can drop the 'Sort Key' column after sorting
df.drop(columns=['Sort Key'], inplace=True)
# Matrices to be plotted
Matrices = ['gmnMAE', 'gmnRMSE']
feature_numbers=[1,2,5,11]
def rename_model(model_name):
    for i in feature_numbers:
        if model_name == f'analysis/linear_1_f{i}':
            return f'{i} feature'
    return model_name  # Returns the original name if no match is found
# Renaming the model names based on the naming convention
df_renamed = df.copy()
df_renamed['Model Name'] = df_renamed['Model Name'].apply(rename_model)
df_renamed = (df_renamed [~df_renamed ['Model Name'].str.contains('linear')])
palette_all = ['#E50068', '#A0006E', '#ffbfeb', '#6E6F73', '#e50900', '#77e500', '#a800e5', '#4d4e51', '#54a000', '#7500a0']
# Redefining the color palette for the renamed model names
updated_palette = {f'{i} feature': color for i, color in zip(feature_numbers, palette_all)}
# updated_palette will now correctly map each feature number to its color
# Extracting relevant columns for each plot
load_columns = ['L0', 'L3', 'L9', 'L11', 'L12', 'L15', 'L16', 'L25', 'L26', 'L32', 'L38', 'L44', 'L45', 'L48', 'L49']
s_column = ['S']
p_column = ['P']
c_column = ['C']

for matrix in Matrices:
    # Filtering the dataframe for the specific metric
    df_filtered = df_renamed[df_renamed['Metric'] == matrix]
    # Preparing data for each plot
    data_load = df_filtered[['Model Name'] + load_columns].melt(id_vars=['Model Name'], var_name='Load', value_name='Value')
    data_s = df_filtered[['Model Name'] + s_column].melt(id_vars=['Model Name'], var_name='Solar', value_name='Value')
    data_p = df_filtered[['Model Name'] + p_column].melt(id_vars=['Model Name'], var_name='Price', value_name='Value')
    data_c = df_filtered[['Model Name'] + c_column].melt(id_vars=['Model Name'], var_name='Carbon', value_name='Value')

    # Plotting
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), gridspec_kw={'width_ratios': [4, 1, 1, 1]})

    # Load plot
    sns.barplot(x='Load', y='Value', hue='Model Name', data=data_load, palette=updated_palette, ax=axes[0])
    axes[0].set_title('Unnormalised Feature - Load')
    axes[0].tick_params(axis='x', rotation=90)
    axes[0].set_ylabel(matrix)
    axes[0].set_xlabel('Building')

    # Solar plot
    sns.barplot(x='Solar', y='Value', hue='Model Name', data=data_s, palette=updated_palette, ax=axes[1])
    axes[1].set_title('Unnormalised Feature - Solar')
    axes[1].get_legend().remove()
    axes[1].tick_params(labelbottom=False)
    axes[1].set_xlabel('')
    axes[1].set_ylabel(matrix)

    # Price plot
    sns.barplot(x='Price', y='Value', hue='Model Name', data=data_p, palette=updated_palette, ax=axes[2])
    axes[2].set_title('Unnormalised Feature - Price')
    axes[2].get_legend().remove()
    axes[2].tick_params(labelbottom=False)
    axes[2].set_xlabel('')
    axes[2].set_ylabel(matrix)

    # Carbon plot
    sns.barplot(x='Carbon', y='Value', hue='Model Name', data=data_c, palette=updated_palette, ax=axes[3])
    axes[3].set_title('Unnormalised Feature - Carbon')
    axes[3].get_legend().remove()
    axes[3].tick_params(labelbottom=False)
    axes[3].set_xlabel('')
    axes[3].set_ylabel(matrix)

    plt.tight_layout()
    plt.savefig('plots/Unnormalised Feature_' + matrix + '.png', dpi=300)
    plt.show()
