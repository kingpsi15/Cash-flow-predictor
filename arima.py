import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
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
model = auto_arima(train,
                   seasonal=True,
                   m=12,
                   start_p=1, start_q=1,
                   max_p=3, max_q=3,
                   start_P=1, start_Q=1,
                   max_P=2, max_Q=2,
                   d=1, D=1,
                   trace=True,
                   error_action='ignore',
                   suppress_warnings=True,
                   stepwise=True)

n_periods = len(test)
forecast = pd.Series(model.predict(n_periods=n_periods), index=test.index)

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
final_model = auto_arima(monthly_net_flow,
                         seasonal=True,
                         m=12,
                         start_p=1, start_q=1,
                         max_p=3, max_q=3,
                         start_P=1, start_Q=1,
                         max_P=2, max_Q=2,
                         d=1, D=1,
                         trace=True,
                         error_action='ignore',
                         suppress_warnings=True,
                         stepwise=True)

future_forecast = pd.Series(final_model.predict(n_periods=6),
                            index=pd.date_range(monthly_net_flow.index[-1] + pd.offsets.MonthEnd(),
                                                periods=6, freq='M'))

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

        model = auto_arima(train,
                           seasonal=True,
                           m=12,
                           start_p=1, start_q=1,
                           max_p=3, max_q=3,
                           start_P=1, start_Q=1,
                           max_P=2, max_Q=2,
                           d=1, D=1,
                           trace=False,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)

        forecast = pd.Series(model.predict(n_periods=len(test)), index=test.index)

        mae = mean_absolute_error(test, forecast)
        rmse = np.sqrt(mean_squared_error(test, forecast))
        print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

        final_model = auto_arima(monthly,
                                 seasonal=True,
                                 m=12,
                                 start_p=1, start_q=1,
                                 max_p=3, max_q=3,
                                 start_P=1, start_Q=1,
                                 max_P=2, max_Q=2,
                                 d=1, D=1,
                                 error_action='ignore',
                                 suppress_warnings=True,
                                 stepwise=True)

        future_forecast = pd.Series(final_model.predict(n_periods=6),
                                    index=pd.date_range(monthly.index[-1] + pd.offsets.MonthEnd(),
                                                        periods=6, freq='M'))

        plt.figure(figsize=(8, 3))
        monthly.plot(label='History')
        future_forecast.plot(label='Forecast', style='--')
        plt.title(f"{category} > {desc}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Skipping {category} > {desc} due to error: {e}")

print(model.summary())
