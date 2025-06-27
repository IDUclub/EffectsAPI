from typing import Optional

import requests
from fastapi import HTTPException
from iduconfig import Config

from app.common.exceptions.http_exception_wrapper import http_exception
from app.dependencies import urban_api_handler

config = Config()

# from app.effects_api.constants.const import URBAN_API


def get_headers(token: Optional[str] = None) -> dict[str, str] | None:
    if token:
        headers = {
            "Authorization": f"Bearer {token}"
        }
        return headers
    return None

async def get_project_id(scenario_id: int, token: Optional[str] = None) -> int:
    endpoint = f"/api/v1/scenarios/{scenario_id}"
    response = await urban_api_handler.get(endpoint, headers={
        "Authorization": f"Bearer {token}"""
    })
    try:
        project_id = response.get("project", {}).get("project_id")
    except Exception:
        raise http_exception(404, "Project ID is missing in scenario data.", scenario_id)

    return project_id

def get_all_project_info(project_id: int, token: Optional[str] = None) -> dict:
    url = config.get("URBAN_API") + f"/api/v1/projects/{project_id}"
    headers = get_headers(token)

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise HTTPException(response.status_code, response.text)
    result = response.json()
    return result

def get_scenario_info(target_scenario_id: int, token) -> dict:

    url = config.get("URBAN_API") + f"/api/v1/scenarios/{target_scenario_id}"
    headers = get_headers(token)
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise HTTPException(response.status_code, response.text)
    return response.json()
