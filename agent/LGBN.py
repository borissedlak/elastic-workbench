import ast
from typing import Dict

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pgmpy.models import LinearGaussianBayesianNetwork

import agent_utils
import utils
from agent.ES_Registry import ServiceType
from agent.SLO_Registry import SLO_Registry
from utils import print_execution_time


class LGBN:
    def __init__(self):
        self.model: LinearGaussianBayesianNetwork = self.init_model()
        self.service_type: ServiceType = None  # TODO: Must set this correctly and use in file collection

    # TODO: This should also fetch files from remote hosts
    # TODO: Also, if I implement multiple service types, this must be considered here
    def collect_all_metric_files(self):
        metrics_local = get_local_metric_file()
        metrics_contents = [metrics_local]
        combined_df = pd.concat([df for _, df in metrics_contents], ignore_index=True)
        return combined_df

    @utils.print_execution_time
    # TODO: The variance is actually important for the uncertainty
    def predict_lgbn_vars(self, partial_state):
        wrapped_in_list = {k: [v] for k, v in partial_state.items()}
        var, mean, vari = self.model.predict(pd.DataFrame(wrapped_in_list))

        samples = {}
        for index, v in enumerate(var):
            mu, sigma = mean[0][index], np.sqrt(vari[index][index])
            sample_val = np.random.normal(mu, sigma, 1)[0]
            samples = samples | {v: int(sample_val)}  # TODO: Might need a different data type at some point

        return samples | partial_state

    def init_model(self):
        df_combined = self.collect_all_metric_files()
        df_cleared = preprocess_data(df_combined)
        return train_lgbn_model(df_cleared)

    def calculate_missing_vars(self, partial_state, assigned_clients: Dict[str, int]):
        full_state = partial_state.copy()

        if "throughput" not in partial_state.keys():
            throughput_expected = (1000 / partial_state['avg_p_latency']) * partial_state['cores']
            full_state = full_state | {"throughput": throughput_expected}

        if "completion_rate" not in partial_state.keys():
            target_throughput = utils.to_absolut_rps(assigned_clients)
            completion_r_expected = full_state['throughput'] / target_throughput * 100
            full_state = full_state | {"completion_rate": completion_r_expected}

        return full_state


def preprocess_data(df):
    df_filtered = agent_utils.filter_rows_during_cooldown(df.copy())
    df_filtered = df_filtered[df_filtered['avg_p_latency'] != -1]  # Filter out rows where we had no processing
    df_filtered.reset_index(drop=True, inplace=True)  # Needed because the filtered does not keep the index

    # Convert and expand service config dict
    df_filtered['s_config'] = df_filtered['s_config'].apply(lambda x: ast.literal_eval(x))
    metadata_expanded = pd.json_normalize(df_filtered['s_config'])

    combined_df_expanded = pd.concat([df_filtered.drop(columns=['s_config']), metadata_expanded], axis=1)
    del combined_df_expanded['timestamp']

    return combined_df_expanded


def get_local_metric_file(path="../share/metrics/metrics.csv"):
    try:
        df = pd.read_csv(path)
        return "local", df
    except Exception as e:
        print(f"Failed to read {path}: {e}")


@print_execution_time  # Roughly 1 to 1.5s
def train_lgbn_model(df, show_result=False):
    # If I don't pass the DAG I have to train it myself, which takes time.
    # scoring_method = AIC(data=df)  # BDeuScore | AICScore
    # estimator = HillClimbSearch(data=df)
    #
    # dag: DAG = estimator.estimate(
    #     scoring_method=scoring_method, max_indegree=5, epsilon=1,
    # )
    # model = LinearGaussianBayesianNetwork(ebunch=dag)
    model = LinearGaussianBayesianNetwork([('pixel', 'avg_p_latency')])  # , ('cores', 'avg_p_latency')])
    # BIFWriter(model).write_bif("./model.xml") # Does not work for LGBNs ....
    model.fit(df)

    for cpd in model.get_cpds():
        print(cpd)

    if show_result:
        for states in [["pixel", "avg_p_latency"]]:  # , ["cores", "avg_p_latency"]]:
            X_samples = model.simulate(1500, 35)
            X_df = pd.DataFrame(X_samples, columns=states)

            sns.jointplot(x=X_df[states[0]], y=X_df[states[1]], kind="kde", height=10, space=0, cmap="viridis")
            plt.show()

    return model


if __name__ == "__main__":
    lgbn = LGBN()
    partial_state_extended = lgbn.predict_lgbn_vars({'pixel': 700, 'cores': 2})
    full_state_expected = lgbn.calculate_missing_vars(partial_state_extended, {"C_1": 100})
    print("Full State", full_state_expected)
    slo_registry = SLO_Registry()

    client_SLOs = slo_registry.get_SLOs_for_client("C_1", ServiceType.QR)
    client_SLO_F_emp = slo_registry.calculate_slo_reward(full_state_expected, client_SLOs)
    print(client_SLO_F_emp)
