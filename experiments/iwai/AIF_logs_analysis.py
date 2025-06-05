import pandas as pd
import matplotlib.pyplot as plt

files = [
    "20250605_104110_pymdp_service_log.csv",
    "20250605_120147_pymdp_service_log.csv",
    "20250605_131211_pymdp_service_log.csv"
]
labels = ["Run 1", "Run 2", "Run 3"]

efe_data = []
info_gain_data = []
pragmatic_value_data = []

for file in files:
    df = pd.read_csv(file, index_col=False)
    print(f"Columns in {file}: {df.columns.tolist()}")
    df = df[["efe", "info_gain", "pragmatic_value"]].copy()
    df.reset_index(drop=True, inplace=True)
    df["timestep"] = range(len(df))
    efe_data.append(df[["timestep", "efe"]])
    info_gain_data.append(df[["timestep", "info_gain"]])
    pragmatic_value_data.append(df[["timestep", "pragmatic_value"]])

# Plot EFE
plt.figure(figsize=(12, 4))
for i, data in enumerate(efe_data):
    plt.plot(data["timestep"], data["efe"], label=labels[i])
plt.xlabel("Timestep")
plt.ylabel("EFE")
plt.legend()
plt.xlim(0, len(efe_data[0]) - 1)
plt.tight_layout()
plt.show()

# Plot Info Gain
plt.figure(figsize=(12, 4))
for i, data in enumerate(info_gain_data):
    plt.plot(data["timestep"], data["info_gain"], label=labels[i])
plt.xlabel("Timestep")
plt.ylabel("Information Gain")
plt.legend()
plt.xlim(0, len(efe_data[0]) - 1)
plt.tight_layout()
plt.show()

# Plot Pragmatic Value
plt.figure(figsize=(12, 4))
for i, data in enumerate(pragmatic_value_data):
    plt.plot(data["timestep"], data["pragmatic_value"], label=labels[i])
plt.xlabel("Timestep")
plt.ylabel("Pragmatic Value")
plt.legend()
plt.xlim(0, len(efe_data[0]) - 1)
plt.tight_layout()
plt.show()