from typing import Dict

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pgmpy.base import DAG
from pgmpy.estimators import AIC, HillClimbSearch
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork

from agent.es_registry import ServiceType
from agent.RRM import calculate_missing_vars, collect_all_metric_files, preprocess_data
from utils import print_execution_time


class LGBN:
    def __init__(self, show_figures=False, structural_training=False, df=None):
        self.show_figures = show_figures
        self.structural_training = structural_training
        self.models: Dict[ServiceType, LinearGaussianBayesianNetwork] = self.init_models(df)

    def init_models(self, df):
        if df is None:  # Remove this df when not needed anymore
            df = collect_all_metric_files()
        df_cleared = preprocess_data(df)
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
        return [('quality', 'throughput'), ('cores', 'throughput')]
    elif service_type == ServiceType.CV:
        return [('cores', 'throughput'), ('model_size', 'throughput'), ('quality', 'throughput')]
    # elif service_type == ServiceType.QR_DEPRECATED:
    #     return [('quality', 'throughput'), ('cores', 'throughput')]
    else:
        raise RuntimeError(f"Service type {service_type} not supported")


if __name__ == "__main__":
    lgbn = LGBN(show_figures=True, structural_training=False)
    print(lgbn.get_linear_relations(ServiceType.CV))
