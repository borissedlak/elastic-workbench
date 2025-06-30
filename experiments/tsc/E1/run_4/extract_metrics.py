import pandas as pd
import ast


# Helper function to construct `s_config` and extract necessary fields
def extract_fields(row):
    state = row["state"]
    s_config = {}

    # Always keep 'data_quality'
    if "data_quality" in state:
        s_config["data_quality"] = int(state["data_quality"])

    # Conditionally add optional keys
    if "model_size" in state:
        s_config["model_size"] = int(state["model_size"])
    if "parallelism" in state:
        s_config["parallelism"] = int(state["parallelism"])
    else:
        s_config["parallelism"] = 1

    return pd.Series({
        "timestamp": row["timestamp"],
        "service_type": row["service"].rsplit("-", 1)[0],
        "container_id": row["service"],
        "avg_p_latency": int(state["avg_p_latency"]) if state["avg_p_latency"] >= 0 else -1,
        "s_config": str(s_config),
        "cores": state["cores"],
        "rps": -1,  # or other formula
        "throughput": int(state.get("throughput", 0)),
        "cooldown": -1  # static
    })

files = [
        'agent_experience_RASK_0_0.1.csv',
             # 'agent_experience_RASK_0_0.05.csv',
             'agent_experience_RASK_0_0.csv',
             'agent_experience_RASK_10_0.1.csv',
             # 'agent_experience_RASK_10_0.05.csv',
             'agent_experience_RASK_10_0.csv',
             'agent_experience_RASK_20_0.1.csv',
             # 'agent_experience_RASK_20_0.05.csv',
             'agent_experience_RASK_20_0.csv'
    ]

for f in files:
    # Load the original CSV file
    df = pd.read_csv(f)

    # Parse the 'state' string to dict
    df["state"] = df["state"].apply(ast.literal_eval)
    suffix= f.replace('agent_experience_RASK_', "").replace(".csv", "")

    # Transform the data
    transformed_df = df.apply(extract_fields, axis=1)

    # Save or display
    transformed_df.to_csv(f"metrics_{suffix}.csv", index=False)
    print(transformed_df)