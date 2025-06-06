import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
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

def evaluate_and_forecast(train, test, seasonal_periods=12):
    results = {}

    # ARIMA model
    try:
        arima_model = auto_arima(train,
                                 seasonal=True,
                                 m=seasonal_periods,
                                 start_p=1, start_q=1,
                                 max_p=3, max_q=3,
                                 start_P=1, start_Q=1,
                                 max_P=2, max_Q=2,
                                 d=1, D=1,
                                 error_action='ignore',
                                 suppress_warnings=True,
                                 stepwise=True)
        arima_forecast = pd.Series(arima_model.predict(n_periods=len(test)), index=test.index)

        arima_mae = mean_absolute_error(test, arima_forecast)
        arima_rmse = np.sqrt(mean_squared_error(test, arima_forecast))

        results['arima'] = {
            'model': arima_model,
            'forecast': arima_forecast,
            'mae': arima_mae,
            'rmse': arima_rmse,
        }
    except Exception as e:
        print("ARIMA failed:", e)
        results['arima'] = None

    # Holt-Winters model
    try:
        hw_model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=seasonal_periods).fit()
        hw_forecast = hw_model.forecast(len(test))
        hw_forecast.index = test.index

        hw_mae = mean_absolute_error(test, hw_forecast)
        hw_rmse = np.sqrt(mean_squared_error(test, hw_forecast))

        results['holtwinters'] = {
            'model': hw_model,
            'forecast': hw_forecast,
            'mae': hw_mae,
            'rmse': hw_rmse,
        }
    except Exception as e:
        print("Holt-Winters failed:", e)
        results['holtwinters'] = None

    return results

# --- Step 4: Evaluate Overall Accuracy and Forecast ---
split_idx = int(len(monthly_net_flow) * 0.8)
train_overall = monthly_net_flow.iloc[:split_idx]
test_overall = monthly_net_flow.iloc[split_idx:]

print("\nEvaluating accuracy on historical data (Overall)...")
overall_results = evaluate_and_forecast(train_overall, test_overall)

# Print accuracy comparison
print("\nOverall Accuracy Comparison:")
for model_name, res in overall_results.items():
    if res:
        print(f"{model_name.upper()} - MAE: {res['mae']:.2f}, RMSE: {res['rmse']:.2f}")

# Plot comparison
plt.figure(figsize=(10, 4))
train_overall.plot(label='Train')
test_overall.plot(label='Actual')

if overall_results['arima']:
    overall_results['arima']['forecast'].plot(label='ARIMA Forecast', style='--')
if overall_results['holtwinters']:
    overall_results['holtwinters']['forecast'].plot(label='Holt-Winters Forecast', style='--')

plt.title("Overall Forecast vs Actual (Test Period)")
plt.legend()
plt.tight_layout()
plt.show()

# --- Step 5: Forecast Future with Both Models ---
print("\nForecasting next 6 months with both models...")

if overall_results['arima']:
    final_arima = auto_arima(monthly_net_flow,
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
    arima_forecast_future = pd.Series(final_arima.predict(n_periods=6),
                                      index=pd.date_range(monthly_net_flow.index[-1] + pd.offsets.MonthEnd(),
                                                          periods=6, freq='M'))
else:
    arima_forecast_future = None

hw_final_model = ExponentialSmoothing(monthly_net_flow, trend='add', seasonal='add', seasonal_periods=12).fit()
hw_forecast_future = hw_final_model.forecast(6)
hw_forecast_future.index = pd.date_range(monthly_net_flow.index[-1] + pd.offsets.MonthEnd(), periods=6, freq='M')

plt.figure(figsize=(10, 4))
monthly_net_flow.plot(label='History')
if arima_forecast_future is not None:
    arima_forecast_future.plot(label='ARIMA Forecast', style='--')
hw_forecast_future.plot(label='Holt-Winters Forecast', style='--')
plt.title("Overall Net Flow Forecast (Next 6 Months)")
plt.legend()
plt.tight_layout()
plt.show()

# --- Step 6: Forecast and Accuracy by Sector/Sub-sector ---
grouped = df.groupby(['Category', 'Description'])

summary_results = []

for (category, desc), group in grouped:
    monthly = group['Net Flow'].resample('ME').sum().dropna()

    if monthly.empty or monthly.sum() == 0 or len(monthly) < 12:
        print(f"Skipping {category} > {desc} due to insufficient data.")
        continue

    print(f"\nEvaluating for {category} > {desc}...")

    split_idx = int(len(monthly) * 0.8)
    train = monthly.iloc[:split_idx]
    test = monthly.iloc[split_idx:]

    res = evaluate_and_forecast(train, test)

    for model_name, metrics in res.items():
        if metrics:
            print(f"{model_name.upper()} - MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}")

    # Plot forecast comparison
    plt.figure(figsize=(8, 3))
    monthly.plot(label='History')
    if res.get('arima'):
        res['arima']['forecast'].plot(label='ARIMA Forecast', style='--')
    if res.get('holtwinters'):
        res['holtwinters']['forecast'].plot(label='Holt-Winters Forecast', style='--')
    plt.title(f"{category} > {desc}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Future forecast for next 6 months
    try:
        future_arima = None
        future_hw = None

        if res['arima']:
            future_arima = auto_arima(monthly,
                                     seasonal=True,
                                     m=12,
                                     start_p=1, start_q=1,
                                     max_p=3, max_q=3,
                                     start_P=1, start_Q=1,
                                     max_P=2, max_Q=2,
                                     d=1, D=1,
                                     error_action='ignore',
                                     suppress_warnings=True,
                                     stepwise=True).predict(n_periods=6)
            future_arima = pd.Series(future_arima,
                                    index=pd.date_range(monthly.index[-1] + pd.offsets.MonthEnd(),
                                                        periods=6, freq='M'))
        hw_final = ExponentialSmoothing(monthly, trend='add', seasonal='add', seasonal_periods=12).fit()
        future_hw = hw_final.forecast(6)
        future_hw.index = pd.date_range(monthly.index[-1] + pd.offsets.MonthEnd(), periods=6, freq='M')

        plt.figure(figsize=(8, 3))
        monthly.plot(label='History')
        if future_arima is not None:
            future_arima.plot(label='ARIMA Forecast', style='--')
        future_hw.plot(label='Holt-Winters Forecast', style='--')
        plt.title(f"{category} > {desc} - Future 6 Months")
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Future forecast failed for {category} > {desc}: {e}")

    # Save summary for comparison
    summary_results.append({
        'Category': category,
        'Description': desc,
        'ARIMA_MAE': res['arima']['mae'] if res['arima'] else None,
        'ARIMA_RMSE': res['arima']['rmse'] if res['arima'] else None,
        'HW_MAE': res['holtwinters']['mae'] if res['holtwinters'] else None,
        'HW_RMSE': res['holtwinters']['rmse'] if res['holtwinters'] else None,
    })

# Summary table of model comparison by sector/sub-sector
summary_df = pd.DataFrame(summary_results)
print("\nSummary of model accuracies by Category and Description:")
print(summary_df)

# You can save this summary to Excel if you want:
# summary_df.to_excel("model_comparison_summary.xlsx", index=False)
