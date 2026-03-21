import pandas as pd
import json

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# reference dataset (training data)
reference_data = pd.read_csv("data/train_clean.csv")

# new production data
current_data = pd.read_csv("data/test_clean.csv")

report = Report(metrics=[
    DataDriftPreset()
])

report.run(
    reference_data=reference_data,
    current_data=current_data
)

report.save_html("monitoring/drift_report.html")
drift_results = report.as_dict()

with open("monitoring/drift_report.json", "w") as f:
    json.dump(drift_results, f, indent=4)

print("Drift report generated")
