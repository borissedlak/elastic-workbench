import ast
from typing import Dict

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pgmpy.base import DAG
from pgmpy.estimators import AIC, HillClimbSearch
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork
from scipy import stats

from agent import agent_utils
from agent.ES_Registry import ServiceType
from utils import print_execution_time


class LGBN:
    def __init__(self, show_figures=False, structural_training=False, df=None):
        self.show_figures = show_figures
        self.structural_training = structural_training
        self.models: Dict[ServiceType, LinearGaussianBayesianNetwork] = self.init_models(df)

    def init_models(self, df):
        if df is None:  # Remove this df when not needed anymore
            df_combined = collect_all_metric_files()
            df_cleared = preprocess_data(df_combined)
        else:
            df_cleared = df

        return train_lgbn_model(df_cleared, self.show_figures, self.structural_training)

    # @utils.print_execution_time
    def predict_lgbn_vars(self, partial_state, service_type: ServiceType, sanitize=True):
        wrapped_in_list = {k: [v] for k, v in partial_state.items()}
        var, mean, vari = self.models[service_type].predict(pd.DataFrame(wrapped_in_list))

        samples = {}
        for index, v in enumerate(var):
            mu, sigma = mean[0][index], np.sqrt(vari[index][index])
            sample_val = np.random.normal(mu, sigma, 1)[0]
            samples = samples | {v: int(sample_val)}

        # TODO: Move somewhere else
        if sanitize:
            for var, min, max in [("avg_p_latency", 1, 999999)]:
                if var in samples.keys():
                    samples[var] = np.clip(samples[var], min, max)

        return partial_state | samples

    def get_expected_state(self, partial_state, service_type: ServiceType, total_rps):
        partial_state_extended = self.predict_lgbn_vars(partial_state, service_type)
        full_state_expected = calculate_missing_vars(partial_state_extended, total_rps)
        return full_state_expected

    def get_linear_relations(self, service_type: ServiceType) -> Dict[str, LinearGaussianCPD]:
        linear_relations = {}
        for cpd in self.models[service_type].get_cpds():
            if not cpd.evidence:  # Only get those relations with dependencies
                continue

            linear_relations[cpd.variable] = cpd
        return linear_relations


# TODO: This is somehow a mess, I wish I could include the replication factor
def calculate_missing_vars(partial_state, total_rps: int):
    full_state = partial_state.copy()

    # TODO: I need to change this formula and remove the #cores as a factor, but include them in the LGBN
    #  Also, I will have to calculate the throughput as the min ((1000 / avg_p), rps)
    if "throughput" not in partial_state.keys():
        # raise RuntimeWarning("Should be included!!")
        throughput_expected = (1000 / partial_state['avg_p_latency']) * partial_state['cores']
        full_state = full_state | {"throughput": throughput_expected}

    if "completion_rate" not in partial_state.keys():
        completion_r_expected = partial_state['throughput'] / total_rps if total_rps > 0 else 1.0
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


# TODO: This should also fetch files from remote hosts
# TODO: Also, if I implement multiple service types, this must be considered here
def collect_all_metric_files():
    metrics_local = get_local_metric_file()
    metrics_contents = [metrics_local]
    combined_df = pd.concat([df for _, df in metrics_contents], ignore_index=True)
    return combined_df


def get_local_metric_file(path="../share/metrics/metrics.csv"):
    try:
        df = pd.read_csv(path)
        return "local", df
    except Exception as e:
        print(f"Failed to read {path}: {e}")


@print_execution_time  # Roughly 1 to 1.5s
def train_lgbn_model(df, show_result=False, structure_training=False):
    service_models = {}

    for service_type_s in df['service_type'].unique():
        df_service = df[df['service_type'] == service_type_s]

        if structure_training:
            del df_service['service_type']
            del df_service['container_id']

            scoring_method = AIC(data=df_service)  # BDeuScore | AICScore
            estimator = HillClimbSearch(data=df_service)

            dag: DAG = estimator.estimate(scoring_method=scoring_method, max_indegree=5)
            model = LinearGaussianBayesianNetwork(ebunch=dag)
        else:
            model = LinearGaussianBayesianNetwork(get_edges_for_service_type(ServiceType(service_type_s)))

        model.fit(df_service)

        for cpd in model.get_cpds():
            print(cpd)

        if show_result:
            for states in [[t[0], t[1]] for t in get_edges_for_service_type(ServiceType(service_type_s))]:
                X_samples = model.simulate(1500, 35)
                X_df = pd.DataFrame(X_samples, columns=states)

                sns.jointplot(x=X_df[states[0]], y=X_df[states[1]], kind="kde", height=10, space=0, cmap="viridis")
                plt.show()
        service_models[ServiceType(service_type_s)] = model

    return service_models


def get_edges_for_service_type(service_type: ServiceType):
    if service_type == ServiceType.QR:
        return [('quality', 'avg_p_latency')]
    elif service_type == ServiceType.CV:
        return [('cores', 'avg_p_latency'), ('model_size', 'avg_p_latency')]
    elif service_type == ServiceType.QR_DEPRECATED:
        return [('quality', 'throughput'), ('cores', 'throughput')]
    else:
        raise RuntimeError(f"Service type {service_type} not supported")


if __name__ == "__main__":
    lgbn = LGBN(show_figures=False, structural_training=False)
    print(lgbn.get_linear_relations(ServiceType.CV))
    # state_expected = lgbn.get_expected_state({'pixel': 700, 'cores': 2}, {"C_1": 100})
    # print("Full State", state_expected)
    # slo_registry = SLO_Registry()
    #
    # client_SLOs = slo_registry.get_SLOs_for_client("C_1", ServiceType.QR)
    # client_SLO_F_emp = slo_registry.calculate_slo_fulfillment(state_expected, client_SLOs)
    # print(client_SLO_F_emp)
