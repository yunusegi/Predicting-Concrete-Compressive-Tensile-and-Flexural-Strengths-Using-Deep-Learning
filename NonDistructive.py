import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Load the dataset
data = pd.read_csv('PaperData.csv')

# Drop rows with any NaN values in the dataset
data = data.dropna()  # Remove rows with NaN values

# Print column names to verify
print("Columns in the dataset:", data.columns)

# Define numerical features (input features)
numerical_features = ['Er', 'UPV']  # Input features

# Function to detect outliers using IQR
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

# Function to remove outliers using IQR
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

# Detect and remove outliers for each numerical feature
outliers_dict = {}
data_cleaned = data.copy()
for feature in numerical_features:
    outliers = detect_outliers_iqr(data, feature)
    outliers_dict[feature] = outliers
    print(f"Outliers in {feature}: {len(outliers)}")
    data_cleaned = remove_outliers_iqr(data_cleaned, feature)

# Verify outlier removal
print(f"Original dataset shape: {data.shape}")
print(f"Cleaned dataset shape (IQR): {data_cleaned.shape}")

# Use the cleaned dataset
data = data_cleaned

# Separate features and target variables
X = data[['Er', 'UPV']]  # Input features
y = data[['Cs', 'Ts', 'Fs']]  # Output targets

# Preprocessing pipeline for numerical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Imputer is redundant now but kept for robustness
    ('scaler', StandardScaler())  # Normalize input features
])

# Combine numerical preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features)  # Only numerical features
    ])

# Apply preprocessing to the features
X_processed = preprocessor.fit_transform(X)

# Normalize the target variables
y_scaler = StandardScaler()
y_processed = y_scaler.fit_transform(y)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=0.2, random_state=50)

# Inverse transform y_train and y_test to original scale for evaluation
y_train_original = y_scaler.inverse_transform(y_train)
y_test_original = y_scaler.inverse_transform(y_test)

# Convert original y_train and y_test to DataFrames for easier indexing
y_train_original = pd.DataFrame(y_train_original, columns=['Cs', 'Ts', 'Fs'])
y_test_original = pd.DataFrame(y_test_original, columns=['Cs', 'Ts', 'Fs'])

# Dictionary to store performance metrics
performance_metrics = {
    'Model': [],
    'Target': [],
    'Metric': [],
    'Value': [],
    'Set': []
}

# Helper function to add metrics to the dictionary
def add_metrics(model_name, y_true, y_pred, set_name):
    targets = ['Cs', 'Ts', 'Fs']
    for i, target in enumerate(targets):
        mae = mean_absolute_error(y_true[target], y_pred[:, i])
        r2 = r2_score(y_true[target], y_pred[:, i])
        performance_metrics['Model'].extend([model_name, model_name])
        performance_metrics['Target'].extend([target, target])
        performance_metrics['Metric'].extend(['MAE', 'R2'])
        performance_metrics['Value'].extend([mae, r2])
        performance_metrics['Set'].extend([set_name, set_name])

# 1. Random Forest Regressor
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=5,
    max_features='sqrt',
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
rf_model.fit(X_train, y_train)

# Predictions for Random Forest
y_pred_rf_test = rf_model.predict(X_test)
y_pred_rf_train = rf_model.predict(X_train)

# Inverse transform predictions
y_pred_rf_test_original = y_scaler.inverse_transform(y_pred_rf_test)
y_pred_rf_train_original = y_scaler.inverse_transform(y_pred_rf_train)

print("\n--- Random Forest Regressor ---")
add_metrics("Random Forest", y_test_original, y_pred_rf_test_original, "Test")
add_metrics("Random Forest", y_train_original, y_pred_rf_train_original, "Train")


# 2. Gradient Boosting Regressor with MultiOutputRegressor
gb_model = MultiOutputRegressor(
    GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.8,
        random_state=42
    )
)
gb_model.fit(X_train, y_train)

# Predictions for Gradient Boosting
y_pred_gb_test = gb_model.predict(X_test)
y_pred_gb_train = gb_model.predict(X_train)

# Inverse transform predictions
y_pred_gb_test_original = y_scaler.inverse_transform(y_pred_gb_test)
y_pred_gb_train_original = y_scaler.inverse_transform(y_pred_gb_train)

print("\n--- Gradient Boosting Regressor ---")
add_metrics("Gradient Boosting", y_test_original, y_pred_gb_test_original, "Test")
add_metrics("Gradient Boosting", y_train_original, y_pred_gb_train_original, "Train")

# 3. Advanced Fully Connected Neural Network
fc_model = Sequential([
    Dense(229, input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.01),

    Dense(229),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.01),

    Dense(229),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.01),

    Dense(229),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.01),

    Dense(229),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.01),
    Dense(y_train.shape[1])
])

optimizer = Adam(learning_rate=0.01)
fc_model.compile(optimizer=optimizer, loss='mse')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.95, patience=5, min_lr=0.0000001)

history = fc_model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=800,
    batch_size=32,
    verbose=0, # Set to 0 to suppress verbose output during training for cleaner console
    callbacks=[reduce_lr]
)

# Predictions for FCNN
y_pred_fc_test_scaled = fc_model.predict(X_test)
y_pred_fc_train_scaled = fc_model.predict(X_train)

# Inverse transform predictions
y_pred_fc_test_original = y_scaler.inverse_transform(y_pred_fc_test_scaled)
y_pred_fc_train_original = y_scaler.inverse_transform(y_pred_fc_train_scaled)

print("\n--- Advanced Fully Connected Neural Network ---")
add_metrics("FCNN", y_test_original, y_pred_fc_test_original, "Test")
add_metrics("FCNN", y_train_original, y_pred_fc_train_original, "Train")

# Create DataFrame for performance metrics
df_performance = pd.DataFrame(performance_metrics)

print("\n--- Performance Metrics (Train and Test) ---")
print(df_performance)

# Plotting Training and Testing Performance

# Plotting MAE
plt.figure(figsize=(15, 7))
sns.barplot(x='Model', y='Value', hue='Set', data=df_performance[df_performance['Metric'] == 'MAE'], palette='viridis')
plt.title('Mean Absolute Error (MAE) for Training and Testing Sets Across Models')
plt.ylabel('MAE')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Plotting R2
plt.figure(figsize=(15, 7))
sns.barplot(x='Model', y='Value', hue='Set', data=df_performance[df_performance['Metric'] == 'R2'], palette='plasma')
plt.title('R-squared (R²) for Training and Testing Sets Across Models')
plt.ylabel('R²')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Function to plot actual vs predicted values with MAE and R² in the title
def plot_actual_vs_predicted(y_test, y_pred, target_col_name, model_name, ax, color, mae, r2):
    ax.scatter(y_test[target_col_name], y_pred[:, y_test.columns.get_loc(target_col_name)], color=color, alpha=0.6)
    min_val = min(y_test[target_col_name].min(), y_pred[:, y_test.columns.get_loc(target_col_name)].min())
    max_val = max(y_test[target_col_name].max(), y_pred[:, y_test.columns.get_loc(target_col_name)].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    ax.set_title(f"{model_name} - Actual vs Predicted ({target_col_name})\nMAE: {mae:.2f}, R²: {r2:.2f}")
    ax.set_xlabel(f'Actual {target_col_name}')
    ax.set_ylabel(f'Predicted {target_col_name}')


# Plot Random Forest results
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
plot_actual_vs_predicted(y_test_original, y_pred_rf_test_original, 'Cs', "Random Forest", axes[0], 'blue',
                         df_performance[(df_performance['Model'] == 'Random Forest') & (df_performance['Target'] == 'Cs') & (df_performance['Metric'] == 'MAE') & (df_performance['Set'] == 'Test')]['Value'].iloc[0],
                         df_performance[(df_performance['Model'] == 'Random Forest') & (df_performance['Target'] == 'Cs') & (df_performance['Metric'] == 'R2') & (df_performance['Set'] == 'Test')]['Value'].iloc[0])
plot_actual_vs_predicted(y_test_original, y_pred_rf_test_original, 'Ts', "Random Forest", axes[1], 'red',
                         df_performance[(df_performance['Model'] == 'Random Forest') & (df_performance['Target'] == 'Ts') & (df_performance['Metric'] == 'MAE') & (df_performance['Set'] == 'Test')]['Value'].iloc[0],
                         df_performance[(df_performance['Model'] == 'Random Forest') & (df_performance['Target'] == 'Ts') & (df_performance['Metric'] == 'R2') & (df_performance['Set'] == 'Test')]['Value'].iloc[0])
plot_actual_vs_predicted(y_test_original, y_pred_rf_test_original, 'Fs', "Random Forest", axes[2], 'green',
                         df_performance[(df_performance['Model'] == 'Random Forest') & (df_performance['Target'] == 'Fs') & (df_performance['Metric'] == 'MAE') & (df_performance['Set'] == 'Test')]['Value'].iloc[0],
                         df_performance[(df_performance['Model'] == 'Random Forest') & (df_performance['Target'] == 'Fs') & (df_performance['Metric'] == 'R2') & (df_performance['Set'] == 'Test')]['Value'].iloc[0])
plt.tight_layout()
plt.show()

# Plot Gradient Boosting results
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
plot_actual_vs_predicted(y_test_original, y_pred_gb_test_original, 'Cs', "Gradient Boosting", axes[0], 'orange',
                         df_performance[(df_performance['Model'] == 'Gradient Boosting') & (df_performance['Target'] == 'Cs') & (df_performance['Metric'] == 'MAE') & (df_performance['Set'] == 'Test')]['Value'].iloc[0],
                         df_performance[(df_performance['Model'] == 'Gradient Boosting') & (df_performance['Target'] == 'Cs') & (df_performance['Metric'] == 'R2') & (df_performance['Set'] == 'Test')]['Value'].iloc[0])
plot_actual_vs_predicted(y_test_original, y_pred_gb_test_original, 'Ts', "Gradient Boosting", axes[1], 'purple',
                         df_performance[(df_performance['Model'] == 'Gradient Boosting') & (df_performance['Target'] == 'Ts') & (df_performance['Metric'] == 'MAE') & (df_performance['Set'] == 'Test')]['Value'].iloc[0],
                         df_performance[(df_performance['Model'] == 'Gradient Boosting') & (df_performance['Target'] == 'Ts') & (df_performance['Metric'] == 'R2') & (df_performance['Set'] == 'Test')]['Value'].iloc[0])
plot_actual_vs_predicted(y_test_original, y_pred_gb_test_original, 'Fs', "Gradient Boosting", axes[2], 'brown',
                         df_performance[(df_performance['Model'] == 'Gradient Boosting') & (df_performance['Target'] == 'Fs') & (df_performance['Metric'] == 'MAE') & (df_performance['Set'] == 'Test')]['Value'].iloc[0],
                         df_performance[(df_performance['Model'] == 'Gradient Boosting') & (df_performance['Target'] == 'Fs') & (df_performance['Metric'] == 'R2') & (df_performance['Set'] == 'Test')]['Value'].iloc[0])
plt.tight_layout()
plt.show()

# Plot FCNN Predictions vs Actual for Cs, Ts, and Fs
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

plot_actual_vs_predicted(y_test_original, y_pred_fc_test_original, 'Cs', "FCNN", axes[0], 'blue',
                         df_performance[(df_performance['Model'] == 'FCNN') & (df_performance['Target'] == 'Cs') & (df_performance['Metric'] == 'MAE') & (df_performance['Set'] == 'Test')]['Value'].iloc[0],
                         df_performance[(df_performance['Model'] == 'FCNN') & (df_performance['Target'] == 'Cs') & (df_performance['Metric'] == 'R2') & (df_performance['Set'] == 'Test')]['Value'].iloc[0])

plot_actual_vs_predicted(y_test_original, y_pred_fc_test_original, 'Ts', "FCNN", axes[1], 'red',
                         df_performance[(df_performance['Model'] == 'FCNN') & (df_performance['Target'] == 'Ts') & (df_performance['Metric'] == 'MAE') & (df_performance['Set'] == 'Test')]['Value'].iloc[0],
                         df_performance[(df_performance['Model'] == 'FCNN') & (df_performance['Target'] == 'Ts') & (df_performance['Metric'] == 'R2') & (df_performance['Set'] == 'Test')]['Value'].iloc[0])

plot_actual_vs_predicted(y_test_original, y_pred_fc_test_original, 'Fs', "FCNN", axes[2], 'green',
                         df_performance[(df_performance['Model'] == 'FCNN') & (df_performance['Target'] == 'Fs') & (df_performance['Metric'] == 'MAE') & (df_performance['Set'] == 'Test')]['Value'].iloc[0],
                         df_performance[(df_performance['Model'] == 'FCNN') & (df_performance['Target'] == 'Fs') & (df_performance['Metric'] == 'R2') & (df_performance['Set'] == 'Test')]['Value'].iloc[0])
plt.tight_layout()
plt.show()

# Plotting FCNN Training Progress (Loss over Epochs)
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('FCNN Training Progress: Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss (Mean Squared Error)')
plt.legend()
plt.grid(True)
plt.show()