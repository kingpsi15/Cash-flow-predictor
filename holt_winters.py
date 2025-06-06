import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# --- Step 1: Load and clean data ---
file_path = r"C:\Users\ADITYA\Downloads\Generated_Cash_Flow_5000_Cleaned.xlsx"  # <- Change this
df = pd.read_excel(file_path)

df = df[(df['Inflow'] != 0) | (df['Outflow'] != 0)].copy()
df['Date'] = pd.to_datetime(df['Date'])
df['Net Flow'] = df['Inflow'] - df['Outflow']
df.sort_values('Date', inplace=True)

print("Cleaned data preview:")
print(df[['Date', 'Category', 'Description', 'Net Flow']].head())

# --- Step 2: Aggregate monthly total ---
df.set_index('Date', inplace=True)
monthly_net_flow = df['Net Flow'].resample('ME').sum()

# --- Step 3: Plot overall trend ---
plt.figure(figsize=(10, 4))
monthly_net_flow.plot(title='Monthly Net Flow (All Sectors)')
plt.ylabel("Net Flow")
plt.tight_layout()
plt.show()

# --- Step 4: Accuracy Check for Overall Net Flow ---
split_idx = int(len(monthly_net_flow) * 0.8)
train = monthly_net_flow.iloc[:split_idx]
test = monthly_net_flow.iloc[split_idx:]

print("\nEvaluating accuracy on historical data (Overall)...")

# Fit Holt-Winters model with additive trend and seasonal component, seasonal period = 12 months
model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12).fit()

n_periods = len(test)
forecast = model.forecast(n_periods)
forecast.index = test.index

mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))
print(f"Overall Accuracy - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# --- Plot Train-Test Forecast ---
plt.figure(figsize=(10, 4))
train.plot(label='Train')
test.plot(label='Actual')
forecast.plot(label='Forecast', style='--')
plt.title("Overall Forecast vs Actual (Test Period)")
plt.legend()
plt.tight_layout()
plt.show()

# --- Step 5: Forecast Future ---
final_model = ExponentialSmoothing(monthly_net_flow, trend='add', seasonal='add', seasonal_periods=12).fit()

future_forecast = final_model.forecast(6)
future_forecast.index = pd.date_range(monthly_net_flow.index[-1] + pd.offsets.MonthEnd(),
                                     periods=6, freq='M')

plt.figure(figsize=(10, 4))
monthly_net_flow.plot(label='History')
future_forecast.plot(label='Forecast', style='--')
plt.title("Overall Net Flow Forecast")
plt.legend()
plt.tight_layout()
plt.show()

# --- Step 6: Forecast and Accuracy by Sector/Sub-sector ---
grouped = df.groupby(['Category', 'Description'])

for (category, desc), group in grouped:
    monthly = group['Net Flow'].resample('ME').sum().dropna()

    if monthly.empty or monthly.sum() == 0 or len(monthly) < 12:
        print(f"Skipping {category} > {desc} due to insufficient data.")
        continue

    print(f"\nEvaluating for {category} > {desc}...")

    try:
        split_idx = int(len(monthly) * 0.8)
        train = monthly.iloc[:split_idx]
        test = monthly.iloc[split_idx:]

        model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12).fit()

        forecast = model.forecast(len(test))
        forecast.index = test.index

        mae = mean_absolute_error(test, forecast)
        rmse = np.sqrt(mean_squared_error(test, forecast))
        print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

        final_model = ExponentialSmoothing(monthly, trend='add', seasonal='add', seasonal_periods=12).fit()

        future_forecast = final_model.forecast(6)
        future_forecast.index = pd.date_range(monthly.index[-1] + pd.offsets.MonthEnd(),
                                             periods=6, freq='M')

        plt.figure(figsize=(8, 3))
        monthly.plot(label='History')
        future_forecast.plot(label='Forecast', style='--')
        plt.title(f"{category} > {desc}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Skipping {category} > {desc} due to error: {e}")

# Optional: print model summary for the last fitted model (Holt-Winters does not have detailed summary like ARIMA)
print("\nLast fitted model parameters:")
print(model.params)
