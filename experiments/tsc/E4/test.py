import pandas as pd

# Load your CSV
df = pd.read_csv("agent_experience_RASK_9_diurnal.csv")

# Column to check the sequence
column = "service"

# Create expected pattern
expected_pattern = ["elastic-workbench-qr-detector-1", "elastic-workbench-cv-analyzer-1", "elastic-workbench-pc-visualizer-1",
                    "elastic-workbench-qr-detector-2", "elastic-workbench-cv-analyzer-2", "elastic-workbench-pc-visualizer-2",
                    "elastic-workbench-qr-detector-3", "elastic-workbench-cv-analyzer-3", "elastic-workbench-pc-visualizer-3"]

# Find violations
violations = []
for i in range(0, len(df), 9):
    chunk = df.iloc[i:i+9]
    if len(chunk) != 9 or not all(chunk[column].values == expected_pattern):
        violations.append(chunk)

# Concatenate and show violations
if violations:
    violations_df = pd.concat(violations)
    print("Violating rows:")
    print(violations_df)
else:
    print("No violations found.")
