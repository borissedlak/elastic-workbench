import ast

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pgmpy.models import LinearGaussianBayesianNetwork

import agent_utils
from utils import print_execution_time


# TODO: This should also fetch files from remote hosts
# TODO: Also, if I implement multiple service types, this must be considered here
def collect_all_metric_files():
    metrics_local = get_local_file()
    metrics_contents = [metrics_local]
    combined_df = pd.concat([df for _, df in metrics_contents], ignore_index=True)
    return combined_df


# TODO: Filter -1 processing_latencies
def preprocess_data(df):
    df_filtered = agent_utils.filter_3s_after_change(df.copy())
    df_filtered = df_filtered[df_filtered['avg_p_latency'] != -1] # Filter out rows where we had no processing
    df_filtered.reset_index(drop=True, inplace=True)  # Needed because the filtered does not keep the index

    # Convert and expand service config dict
    df_filtered['s_config'] = df_filtered['s_config'].apply(lambda x: ast.literal_eval(x))
    metadata_expanded = pd.json_normalize(df_filtered['s_config'])

    combined_df_expanded = pd.concat([df_filtered.drop(columns=['s_config']), metadata_expanded], axis=1)
    # print(combined_df_expanded)

    return combined_df_expanded


def get_local_file(path="../share/metrics/metrics.csv"):
    try:
        # Read CSV content into a DataFrame
        df = pd.read_csv(path)
        # Append a tuple of filename and content
        return "local", df
    except Exception as e:
        print(f"Failed to read {path}: {e}")


@print_execution_time  # Roughly 1 to 1.5s
def train_lgbn_model(df, show_result=False):
    # If I don't pass the DAG I have to train it myself, which takes time.
    # scoring_method = AICScore(data=df_filtered)  # BDeuScore | AICScore
    # estimator = HillClimbSearch(data=df_filtered)
    #
    # dag: pgmpy.base.DAG = estimator.estimate(
    #     scoring_method=scoring_method, max_indegree=5, epsilon=1,
    # )
    # model = LinearGaussianBayesianNetwork(ebunch=dag)
    model = LinearGaussianBayesianNetwork([('pixel', 'fps'), ('cores', 'fps')])
    # XMLBIFWriter(model).write_xmlbif("../model.xml")
    model.fit(df)

    for cpd in model.get_cpds():
        print(cpd)

    if show_result:
        for states in [["pixel", "fps"], ["cores", "fps"]]:
            X_samples = model.simulate(1500, 35)
            X_df = pd.DataFrame(X_samples, columns=states)

            sns.jointplot(x=X_df[states[0]], y=X_df[states[1]], kind="kde", height=10, space=0, cmap="viridis")
            plt.show()

    return model


if __name__ == "__main__":
    df_combined = collect_all_metric_files()
    df_cleared = preprocess_data(df_combined)
    train_lgbn_model(df_cleared, show_result=True)
