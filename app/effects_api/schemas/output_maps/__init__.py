import json
from pathlib import Path

abs_path = Path(__file__).parent


with open(
    abs_path / "soc_economy_pred_name_map.json", "r", encoding="utf-8"
) as sepnm_file:
    soc_economy_pred_name_map = json.load(sepnm_file)

with open(abs_path / "pred_columns_names_map.json", "r", encoding="utf-8") as pcnp_file:
    pred_columns_names_map = json.load(pcnp_file)
