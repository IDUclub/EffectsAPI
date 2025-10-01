from typing import Any, Dict, Optional


def _truthy_is_based(v: Any) -> bool:
    return v is True or v == 1 or (isinstance(v, str) and v.lower() == "true")


def _parent_id(s: Dict[str, Any]) -> Optional[int]:
    p = s.get("parent_scenario")
    return p.get("id") if isinstance(p, dict) else p


def _sid(s: Dict[str, Any]) -> Optional[int]:
    try:
        return int(s.get("scenario_id"))
    except Exception:
        return None


async def _resolve_base_id(urban_api_client, token: str, scenario_id: int) -> int:
    info = await urban_api_client.get_scenario_info(scenario_id, token)
    project_id = (info.get("project") or {}).get("project_id")
    regional_id = (info.get("parent_scenario") or {}).get("id")

    if not project_id or not regional_id:
        return scenario_id

    scenarios = await urban_api_client.get_project_scenarios(project_id, token)
    matches = [
        s for s in scenarios
        if _truthy_is_based(s.get("is_based")) and _parent_id(s) == regional_id and _sid(s) is not None
    ]
    if not matches:
        only_based = [s for s in scenarios if _truthy_is_based(s.get("is_based")) and _sid(s) is not None]
        if not only_based:
            return scenario_id
        matches = only_based

    matches.sort(key=lambda x: (x.get("updated_at") is not None, x.get("updated_at")), reverse=True)
    return _sid(matches[0]) or scenario_id
