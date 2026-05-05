import pandas as pd

def split_reps(input_file):
    df = pd.read_csv(input_file)
    
    first_half_rows = []
    second_half_rows = []
    
    # Iterate through each repetition group
    for rep_id, group in df.groupby('rep'):
        # Find the 50% mark for this specific group
        midpoint = len(group) // 2
        
        # Slice the group and store
        first_half_rows.append(group.iloc[:midpoint])
        second_half_rows.append(group.iloc[midpoint:])
    
    # Reassemble and save
    pd.concat(first_half_rows).to_csv('agent_experience_EXPLORE.csv', index=False)
    pd.concat(second_half_rows).to_csv('agent_experience_EXPLOIT.csv', index=False)

split_reps('agent_experience_RASK_180_0.csv')