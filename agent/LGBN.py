import ast
from typing import Dict

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pgmpy.models import LinearGaussianBayesianNetwork
from scipy import stats

import agent_utils
import utils
from agent.ES_Registry import ServiceType
from utils import print_execution_time


class LGBN:
    def __init__(self, show_figures=False):
        self.show_figures = show_figures
        self.models: Dict[str, LinearGaussianBayesianNetwork] = self.init_models()
        self.service_type: ServiceType = None  # TODO: Must set this correctly and use in file collection

    def init_models(self):
        df_combined = self.collect_all_metric_files()
        df_cleared = preprocess_data(df_combined)
        return train_lgbn_model(df_cleared, show_result=self.show_figures)

    # TODO: This should also fetch files from remote hosts
    # TODO: Also, if I implement multiple service types, this must be considered here
    def collect_all_metric_files(self):
        metrics_local = get_local_metric_file()
        metrics_contents = [metrics_local]
        combined_df = pd.concat([df for _, df in metrics_contents], ignore_index=True)
        return combined_df

    @utils.print_execution_time
    def predict_lgbn_vars(self, partial_state, sanitize=True):
        wrapped_in_list = {k: [v] for k, v in partial_state.items()}
        var, mean, vari = self.models.predict(pd.DataFrame(wrapped_in_list))

        samples = {}
        for index, v in enumerate(var):
            mu, sigma = mean[0][index], np.sqrt(vari[index][index])
            sample_val = np.random.normal(mu, sigma, 1)[0]
            samples = samples | {v: int(sample_val)}  # TODO: Might need a different data type at some point

        # TODO: Move somewhere else
        if sanitize:
            for var, min, max in [("avg_p_latency", 1, 999999)]:
                if var in samples.keys():
                    samples[var] = np.clip(samples[var], min, max)

        return partial_state | samples

    def get_expected_state(self, partial_state, assigned_clients):
        partial_state_extended = self.predict_lgbn_vars(partial_state)
        full_state_expected = calculate_missing_vars(partial_state_extended, assigned_clients)
        return full_state_expected

    def get_linear_relations(self, service_type: ServiceType):
        linear_relations = {}
        for cpd in self.models[service_type.value].get_cpds():
            if cpd.evidence == []:  # Only get those relations with dependencies
                continue

            # TODO: This I will need to fix when I get more variables
            linear_relations[cpd.variable] = [(cpd.variables[1], cpd.beta[1], cpd.beta[0])]
        return linear_relations


def calculate_missing_vars(partial_state, assigned_clients: Dict[str, int]):
    full_state = partial_state.copy()

    if "throughput" not in partial_state.keys():
        throughput_expected = (1000 / partial_state['avg_p_latency']) * partial_state['cores']
        full_state = full_state | {"throughput": throughput_expected}

    if "completion_rate" not in partial_state.keys():
        target_throughput = utils.to_absolut_rps(assigned_clients)
        completion_r_expected = full_state['throughput'] / target_throughput
        full_state = full_state | {"completion_rate": completion_r_expected}

    return full_state


def preprocess_data(df):
    df_filtered = agent_utils.filter_rows_during_cooldown(df.copy())
    df_filtered = df_filtered[df_filtered['avg_p_latency'] >= 0]  # Filter out rows where we had no processing
    z_scores = np.abs(stats.zscore(df_filtered['avg_p_latency']))
    df_filtered = df_filtered[z_scores < 2.0]  # 3 is a common threshold for extreme outliers
    df_filtered.reset_index(drop=True, inplace=True)  # Needed because the filtered does not keep the index

    # Convert and expand service config dict
    df_filtered['s_config'] = df_filtered['s_config'].apply(lambda x: ast.literal_eval(x))
    metadata_expanded = pd.json_normalize(df_filtered['s_config'])

    combined_df_expanded = pd.concat([df_filtered.drop(columns=['s_config']), metadata_expanded], axis=1)
    del combined_df_expanded['timestamp']

    print(f"Training data contains service types {df_filtered['service_type'].unique()}")

    return combined_df_expanded


def get_local_metric_file(path="../share/metrics/metrics.csv"):
    try:
        df = pd.read_csv(path)
        return "local", df
    except Exception as e:
        print(f"Failed to read {path}: {e}")


@print_execution_time  # Roughly 1 to 1.5s
def train_lgbn_model(df, show_result=False):
    service_models = {}

    for service_type in df['service_type'].unique():
        model = get_lgbn_for_service_type(ServiceType(service_type))
        df_service = df[df['service_type'] == service_type]
        model.fit(df_service)

        for cpd in model.get_cpds():
            print(cpd)

        if show_result:
            for states in [["quality", "avg_p_latency"]]:  # , ["cores", "avg_p_latency"]]:
                X_samples = model.simulate(1500, 35)
                X_df = pd.DataFrame(X_samples, columns=states)

                sns.jointplot(x=X_df[states[0]], y=X_df[states[1]], kind="kde", height=10, space=0, cmap="viridis")
                plt.show()
        service_models[service_type] = model

    return service_models


def get_lgbn_for_service_type(service_type: ServiceType):
    if service_type == ServiceType.QR:
        return LinearGaussianBayesianNetwork([('quality', 'avg_p_latency')])  # , ('cores', 'avg_p_latency')])
    elif service_type == ServiceType.CV:
        return LinearGaussianBayesianNetwork([('quality', 'avg_p_latency'), ('cores', 'avg_p_latency'), ('model_size', 'avg_p_latency')])
    else:
        raise RuntimeError(f"Service type {service_type} not supported")


if __name__ == "__main__":
    lgbn = LGBN(show_figures=True)
    print(lgbn.get_linear_relations(ServiceType.QR))
    # state_expected = lgbn.get_expected_state({'pixel': 700, 'cores': 2}, {"C_1": 100})
    # print("Full State", state_expected)
    # slo_registry = SLO_Registry()
    #
    # client_SLOs = slo_registry.get_SLOs_for_client("C_1", ServiceType.QR)
    # client_SLO_F_emp = slo_registry.calculate_slo_fulfillment(state_expected, client_SLOs)
    # print(client_SLO_F_emp)
