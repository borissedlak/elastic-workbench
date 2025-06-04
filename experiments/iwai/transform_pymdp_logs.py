import os

import pandas as pd

from agent.SLORegistry import calculate_slo_fulfillment, SLO_Registry, to_normalized_slo_f
from agent.agent_utils import FullStateDQN
from agent.es_registry import ServiceType

FullStateDQN(None, None, None, None, None, None, None, None, None)  # Keep here so the type is known

def import_pymdp_logs(filename):
    df = pd.read_csv(filename)

    ROOT = os.path.dirname(__file__)
    slo_registry = SLO_Registry(ROOT + "/../../config/slo_config.json")
    client_slos_qr = slo_registry.get_all_SLOs_for_assigned_clients(
        ServiceType.QR, {"C_1": 100}
    )[0]
    client_slos_cv = slo_registry.get_all_SLOs_for_assigned_clients(
        ServiceType.CV, {"C_1": 100}
    )[0]

    rows = []
    rep = 1  # assuming rep is always 1

    for _, row in df.iterrows():
        timestamp = row['timestamp']

        # Process QR service
        qr_state_obj = eval(row['next_state_qr'])  # or use literal_eval with a parser if needed
        qr_state = qr_state_obj._asdict()
        # qr_slo_f = to_normalized_slo_f(
        #             calculate_slo_fulfillment(qr_state_obj.to_normalized_dict(), client_slos_qr),
        #             client_slos_qr,
        #         )
        qr_slo_f = row['reward'] / 2
        # qr_slo_f += 0.1 if row['action_qr'] == "DILLY_DALLY" else 0.0
        rows.append({
            "rep": rep,
            "timestamp": timestamp,
            "service": "elastic-workbench-qr-detector-1",
            "slo_f": qr_slo_f,
            "state": str(qr_state)
        })

        # Process CV service
        cv_state_obj = eval(row['next_state_cv'])
        cv_state = cv_state_obj._asdict()
        # cv_slo_f = to_normalized_slo_f(
        #             calculate_slo_fulfillment(cv_state_obj.to_normalized_dict(), client_slos_cv),
        #             client_slos_cv,
        #         )
        cv_slo_f = row['reward'] / 2
        # cv_slo_f += 0.1 if row['action_cv'] == "DILLY_DALLY" else 0.0
        rows.append({
            "rep": rep,
            "timestamp": timestamp,
            "service": "elastic-workbench-cv-analyzer-1",
            "slo_f": cv_slo_f,
            "state": str(cv_state)
        })

    # Output DataFrame
    output_df = pd.DataFrame(rows)[:80] # As for the other two agents
    output_df.to_csv(ROOT + "/B1/agent_experience_AIF.csv", index=False)

if __name__ == "__main__":
    import_pymdp_logs(filename = "20250604_165223_pymdp_service_log.csv")