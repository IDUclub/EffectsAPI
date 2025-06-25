from typing import Optional

import requests
from fastapi import HTTPException

from app.effects_api.constants.const import URBAN_API


def get_headers(token: Optional[str] = None) -> dict[str, str] | None:
    if token:
        headers = {
            "Authorization": f"Bearer {token}"
        }
        return headers
    return None

def get_project_id(scenario_id: int, token: Optional[str] = None) -> int:
    url = f"{URBAN_API}/api/v1/scenarios/{scenario_id}"
    headers = get_headers(token)

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise HTTPException(response.status_code, response.text)
    return response.json()["project"]["project_id"]

def get_all_project_info(project_id: int, token: Optional[str] = None) -> dict:
    url = f"{URBAN_API}/api/v1/projects/{project_id}"
    headers = get_headers(token)

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise HTTPException(response.status_code, response.text)
    result = response.json()
    return result

def get_scenario_info(target_scenario_id: int, token) -> dict:

    url = f"{URBAN_API}/api/v1/scenarios/{target_scenario_id}"
    headers = get_headers(token)
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise HTTPException(response.status_code, response.text)
    return response.json()
