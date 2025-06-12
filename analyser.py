"""
Project: Resistance Drift Analysis using Impedance Measurements
Author: Aman Malhotra
Organization: CSIR-National Physical Laboratory (CSIR-NPL)
Frequency Analyzed: 1 kHz

Description:
This script was developed as part of the CSIR-NPL Impedance Analyser Internship Project.
It performs data cleaning, statistical analysis and visualization of resistance measurements
recorded over multiple days under varying environmental conditions (temperature & humidity).

Key features include:
- Summary statistics and trend analysis
- Allan deviation computation
- Drift analysis over time and vs. environment
- Visual output in HTML-friendly format

All plots are auto-generated and saved to a specified directory for integration with a web dashboard.
"""

# === Importing Required Libraries ===
import pandas as pd  # for reading and processing data
import numpy as np   # for numerical operations
import matplotlib.pyplot as plt  # for plotting
import seaborn as sns  # for advanced statistical plots
import os  # for directory handling

# === Step 0: Create output directory for saving plots ===
save_dir = os.path.join("plots")
os.makedirs(save_dir, exist_ok=True)  # create if not exists

# === Step 1: Define file paths with associated dates ===
file_paths = {
    '2024-07-26': os.path.join("data", "26July2024.csv"),
    '2024-07-29': os.path.join("data", "29July2024.csv"),
    '2024-07-30': os.path.join("data", "30july2024.csv"),
    '2024-07-31': os.path.join("data", "31july2024.csv"),
}

all_batches = []  # to store cleaned data from each file

# === Step 2: Load, clean, and standardize each file ===
for date, path in file_paths.items():
    try:
        df = pd.read_csv(path, skip_blank_lines=True)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        continue

    # Remove completely empty columns and standardize column names
    df = df.dropna(axis=1, how='all')
    df.columns = [col.strip().upper() for col in df.columns]

    # Map varied column names to standard ones
    col_map = {}
    for col in df.columns:
        if col == 'R/H' or 'HUM' in col:
            col_map[col] = 'HUMIDITY'
        elif 'TEMP' in col:
            col_map[col] = 'TEMPERATURE'
        elif 'RES' in col:
            col_map[col] = 'RESISTANCE'
    df = df.rename(columns=col_map)

    # Check if all required columns are present
    required_cols = {'HUMIDITY', 'TEMPERATURE', 'RESISTANCE'}
    if not required_cols.issubset(df.columns):
        print(f"Skipping {path}: Missing required columns.")
        continue

    # Keep only relevant columns and drop any rows with NaN
    df = df[['HUMIDITY', 'TEMPERATURE', 'RESISTANCE']].dropna()
    df['DATE'] = date  # Add date column
    all_batches.append(df)

# === Step 3: Combine data from all dates into one DataFrame ===
full_df = pd.concat(all_batches, ignore_index=True)
full_df[['HUMIDITY', 'TEMPERATURE', 'RESISTANCE']] = full_df[['HUMIDITY', 'TEMPERATURE', 'RESISTANCE']].astype(float)

# === Step 4: Compute summary statistics (mean, std, count) by date ===
summary = full_df.groupby('DATE').agg({
    'RESISTANCE': ['mean', 'std', 'count'],
    'TEMPERATURE': 'mean',
    'HUMIDITY': 'mean'
}).reset_index()

# Flatten multi-level column names
summary.columns = ['Date', 'Mean_Resistance', 'Std_Resistance', 'Count', 'Mean_Temperature', 'Mean_Humidity']

# === Step 4B: Visualize summary statistics ===

# Plot 1: Mean Resistance ± Std Deviation
plt.figure(figsize=(10, 5))
plt.errorbar(summary['Date'], summary['Mean_Resistance'], yerr=summary['Std_Resistance'],
             fmt='-o', capsize=5, color='darkblue', label='Mean Resistance ± Std Dev')
plt.title('Mean Resistance Over Time')
plt.xlabel('Date')
plt.ylabel('Resistance (Ohms)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'Summary_Mean_Resistance.png'))
plt.close()

# Plot 2: Mean Temperature Over Time
plt.figure(figsize=(10, 4))
sns.lineplot(x='Date', y='Mean_Temperature', data=summary, marker='o', color='orange')
plt.title('Mean Temperature Over Time')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'Summary_Mean_Temperature.png'))
plt.close()

# Plot 3: Mean Humidity Over Time
plt.figure(figsize=(10, 4))
sns.lineplot(x='Date', y='Mean_Humidity', data=summary, marker='o', color='teal')
plt.title('Mean Humidity Over Time')
plt.xlabel('Date')
plt.ylabel('Humidity (%)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'Summary_Mean_Humidity.png'))
plt.close()

# === Step 5: Define Allan Deviation Function ===
def allan_deviation(data, max_m=200):
    """
    Compute Allan deviation for a time-series data vector.
    - data: 1D numpy array (resistance values)
    - max_m: max number of averaging intervals
    Returns:
    - taus: list of averaging times
    - adevs: list of corresponding Allan deviation values
    """
    N = len(data)
    max_m = min(max_m, N // 2)
    taus, adevs = [], []

    cumsum = np.cumsum(data)
    for m in range(1, max_m + 1):
        avgs = (cumsum[m:] - cumsum[:-m]) / m
        diff = avgs[1:] - avgs[:-1]
        allan_var = 0.5 * np.mean(diff**2)
        adev = np.sqrt(allan_var)

        taus.append(m)
        adevs.append(adev)

    return np.array(taus), np.array(adevs)

# === Step 6: Plot Allan Deviation for Each Date ===
plt.figure(figsize=(12, 6))
for date in summary['Date']:
    res_data = full_df[full_df['DATE'] == date]['RESISTANCE'].values
    if len(res_data) < 3:
        continue
    taus, adevs = allan_deviation(res_data)
    plt.loglog(taus, adevs, label=date)

plt.xlabel('Averaging Time (samples)')
plt.ylabel('Allan Deviation (Ohms)')
plt.title('Allan Deviation of Resistance Measurements Over Time')
plt.grid(True, which='both', ls='--')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'Allan_Deviation_Comparison.png'))
plt.close()

# === Step 7: Resistance Drift Over Time ===
plt.figure(figsize=(10, 5))
plt.errorbar(summary['Date'], summary['Mean_Resistance'], yerr=summary['Std_Resistance'],
             fmt='-o', capsize=5, color='darkred')
plt.title('Resistance Drift Over Time @ 1 kHz')
plt.xlabel('Date')
plt.ylabel('Mean Resistance (Ohms)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'Resistance_Drift_Over_Time.png'))
plt.close()

# === Step 8: Identify Common Environmental Conditions ===
TEMP_TOL = 0.5  # °C tolerance
HUM_TOL = 1.0   # %RH tolerance

env_df = full_df[['DATE', 'TEMPERATURE', 'HUMIDITY', 'RESISTANCE']].copy()
env_df['TEMP_ROUND'] = env_df['TEMPERATURE'].round(1)
env_df['HUM_ROUND'] = env_df['HUMIDITY'].round(1)

grouped = env_df.groupby(['TEMP_ROUND', 'HUM_ROUND'])
common_conditions = []

# Find conditions that appear on more than one date
for (temp_r, hum_r), group in grouped:
    unique_dates = group['DATE'].unique()
    if len(unique_dates) > 1:
        common_conditions.append((temp_r, hum_r, unique_dates))

print(f"Found {len(common_conditions)} environmental conditions appearing on multiple dates.")

# === Step 9: Plot Resistance Drift vs Temperature and Humidity ===
drift_temp_df = summary.copy()

# Compute % drift relative to first date
baseline_res = drift_temp_df['Mean_Resistance'].iloc[0]
drift_temp_df['Resistance_Drift (%)'] = 100 * (drift_temp_df['Mean_Resistance'] - baseline_res) / baseline_res

# Plot Drift vs Temperature
plt.figure(figsize=(8, 5))
sns.scatterplot(data=drift_temp_df, x='Mean_Temperature', y='Resistance_Drift (%)',
                hue='Date', palette='tab10', s=100, edgecolor='black')
sns.regplot(data=drift_temp_df, x='Mean_Temperature', y='Resistance_Drift (%)',
            scatter=False, color='red', line_kws={"linewidth": 1.5})
plt.title('Resistance Drift vs Temperature')
plt.xlabel('Mean Temperature (°C)')
plt.ylabel('Resistance Drift (%)')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'Drift_vs_Temperature.png'))
plt.close()

# Plot Drift vs Humidity
plt.figure(figsize=(8, 5))
sns.scatterplot(data=drift_temp_df, x='Mean_Humidity', y='Resistance_Drift (%)',
                hue='Date', palette='tab10', s=100, edgecolor='black')
sns.regplot(data=drift_temp_df, x='Mean_Humidity', y='Resistance_Drift (%)',
            scatter=False, color='green', line_kws={"linewidth": 1.5})
plt.title('Resistance Drift vs Humidity')
plt.xlabel('Mean Humidity (%)')
plt.ylabel('Resistance Drift (%)')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'Drift_vs_Humidity.png'))
plt.close()

# === Step 10: Boxplots for Resistance under Common Conditions ===
N = 5  # limit to 5 plots
for i, (temp_r, hum_r, dates) in enumerate(common_conditions[:N]):
    cond_df = env_df[(np.isclose(env_df['TEMP_ROUND'], temp_r, atol=TEMP_TOL)) &
                     (np.isclose(env_df['HUM_ROUND'], hum_r, atol=HUM_TOL)) &
                     (env_df['DATE'].isin(dates))]

    plt.figure(figsize=(6, 5))
    sns.boxplot(x='DATE', y='RESISTANCE', data=cond_df)
    plt.title(f'Temp ≈ {temp_r}°C, Hum ≈ {hum_r}%')
    plt.ylabel('Resistance (Ohms)')
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.tight_layout()

    temp_str = str(temp_r).replace('.', '_')
    hum_str = str(hum_r).replace('.', '_')
    filename = f'Resistance_Comparison_{i+1}_Temp{temp_str}_Hum{hum_str}.png'
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()
