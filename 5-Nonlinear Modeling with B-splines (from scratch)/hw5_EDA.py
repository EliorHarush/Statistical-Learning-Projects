import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the training data
train_df = pd.read_csv('hw5_train_s1467.csv')

# Printing basic information and summary statistics
print("--- Dataset Info ---")
train_df.info()

print("\n--- Summary Statistics ---")
print(train_df.describe())

# Setting up the visual grid
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot A: The Non-Linear Core (Temperature vs. Consumption)
# Highlighting the High Demand Alert to see where the thresholds lie
sns.scatterplot(
    data=train_df, 
    x='temperature', 
    y='consumption_kwh', 
    hue='high_demand_alert', 
    palette='coolwarm', 
    ax=axes[0, 0]
)
axes[0, 0].set_title('Temperature vs. Consumption (The U-Shape)')
axes[0, 0].set_xlabel('Temperature (°C)')
axes[0, 0].set_ylabel('Consumption (kWh)')

# Plot B: Humidity vs. Consumption
sns.scatterplot(
    data=train_df, 
    x='humidity', 
    y='consumption_kwh', 
    hue='high_demand_alert',
    palette='coolwarm',
    ax=axes[0, 1]
)
axes[0, 1].set_title('Humidity vs. Consumption')
axes[0, 1].set_xlabel('Humidity (%)')
axes[0, 1].set_ylabel('Consumption (kWh)')

# Plot C: Weekend Baseline Shift
sns.boxplot(
    data=train_df, 
    x='weekend', 
    y='consumption_kwh', 
    palette='Set2', 
    ax=axes[1, 0]
)
axes[1, 0].set_title('Weekend vs. Consumption')
axes[1, 0].set_xlabel('Weekend (0 = Weekday, 1 = Weekend)')
axes[1, 0].set_ylabel('Consumption (kWh)')

# Plot D: Temperature Distribution by Alert Status
sns.histplot(
    data=train_df, 
    x='temperature', 
    hue='high_demand_alert', 
    multiple='stack', 
    palette='coolwarm', 
    bins=20, 
    ax=axes[1, 1]
)
axes[1, 1].set_title('Temperature Distribution by High Demand Alert')
axes[1, 1].set_xlabel('Temperature (°C)')

plt.tight_layout()
plt.show()