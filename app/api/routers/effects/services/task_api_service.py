from datetime import datetime
from typing import Optional

import requests
from fastapi import HTTPException

from app.api.utils.const import URBAN_API
from app.api.routers.effects.task_schema import TaskInfoSchema



def get_headers(token: Optional[str] = None) -> dict[str, str] | None:
    if token:
        headers = {
            "Authorization": f"Bearer {token}"
        }
        return headers
    return None

def get_project_id(scenario_id: int, token: Optional[str] = None) -> int:
    url = f"{URBAN_API}/api/scenarios/{scenario_id}"
    headers = get_headers(token)

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise HTTPException(response.status_code, response.text)
    return response.json()["project"]["project_id"]

def get_project_info(project_id: int, token: Optional[str] = None) -> TaskInfoSchema:
    url = f"{URBAN_API}/api/projects/{project_id}"
    headers = get_headers(token)

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise HTTPException(response.status_code, response.text)
    result = response.json()
    if result.get("updated_at"):
        lust_update = datetime.strptime(result["updated_at"], "%Y-%m-%dT%H:%M:%S.%fZ")
    else:
        lust_update = datetime.strptime(result["created_at"], "%Y-%m-%dT%H:%M:%S.%fZ")
    project_info = TaskInfoSchema(
        project_id=project_id,
        base_scenario_id=result["base_scenario"]["id"],
        lust_update=lust_update
    )
    return project_info
