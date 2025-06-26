import json
import os
from datetime import datetime
from typing import Annotated

from loguru import logger
# from blocksnet.models import ServiceType
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from starlette.responses import JSONResponse
import geopandas as gpd

from app.effects_api.constants import const
from app.common import auth, decorators
from app.effects_api.models import effects_models as em
from app.effects_api.modules import effects_service as es, service_type_service as sts
from app.effects_api.modules.f22_service import run_development_parameters
from app.effects_api.modules.scenario_service import get_scenario_functional_zones
from app.effects_api.schemas.task_schema import TaskSchema, TaskStatusSchema, TaskInfoSchema
from app.effects_api.modules.task_api_service import get_scenario_info, get_all_project_info, get_project_id

router = APIRouter(prefix='/effects', tags=['Effects'])

def on_startup(): # TODO оценка базовых сценариев
    if not os.path.exists(const.DATA_PATH):
        logger.info(f'Creating data folder at {const.DATA_PATH}')
        os.mkdir(const.DATA_PATH)

tasks: dict[int, TaskSchema] = {}

def check_task_evaluation(scenario_id: int) -> None:

    if not tasks.get(scenario_id):
        raise HTTPException(
            404,
            detail={
                "msg": f"Calculations for scenario {scenario_id} was never started",
                "detail": {
                    "available scenarios": list(tasks.keys())
                }
            }

        )
    elif tasks[scenario_id].task_status.task_status == "pending":
        raise HTTPException(
            400,
            detail={
                "msg": f"Calculations for scenario {scenario_id} is still running",
                "detail": {
                    "available results": [ i for i in tasks.values() if i.task_status.task_status == "success" ],
                }
            }
        )

    elif tasks[scenario_id].task_status == "error":
        raise HTTPException(
            500,
            detail={
                "msg": f"Calculations for scenario {scenario_id} failed",
                "detail": {
                    "error": tasks[scenario_id].task_status.task_status,
                }
            }
        )
    elif tasks[scenario_id].task_status.task_status == "success":
        return
    else:
        raise HTTPException(
            500,
            detail={
                "msg": f"Unexpected error during task check",
                "detail": {
                    "unknown status": tasks[scenario_id].task_status.task_status,
                }
            }
        )


#ToDo rewrite to check token firstly
def check_or_set_status(project_scenario_id: int, token) -> dict:

    scenario_info = get_scenario_info(project_scenario_id, token)

    if task_info := tasks.get(project_scenario_id):
        task_date = task_info.task_info.lust_update
        if scenario_info.get("updated_at"):
            actual_date = datetime.strptime(scenario_info["updated_at"], "%Y-%m-%dT%H:%M:%S.%fZ")
        else:
            actual_date = datetime.strptime(scenario_info["created_at"], "%Y-%m-%dT%H:%M:%S.%fZ")
        if actual_date > task_date:
            task_info.task_info.lust_update = actual_date
            task_info.task_status.task_status = "pending"
            return {"action": "continue"}
        match task_info.task_status.task_status:
            case "success":
                return {
                    "action": "return",
                    "msg": "task is already done and up to date",
                    "task_info": task_info,
                }
            case "pending":
                return {
                    "action": "return",
                    "msg": "task is already running",
                    "task_info": task_info,
                }
            case"done":
                return {
                    "action": "return",
                    "msg": "task is done",
                    "task_info": task_info,
                }
            case "error":
                return {
                    "action": "return",
                    "msg": "task failed due to error",
                    "task_info": task_info,
                }
            case _:
                raise HTTPException(status_code=500, detail="Unknown task status")
    else:
        project_id = get_project_id(project_scenario_id, token)
        project_info = get_all_project_info(project_id, token)
        if scenario_info.get("updated_at"):
            lust_update = datetime.strptime(scenario_info["updated_at"], "%Y-%m-%dT%H:%M:%S.%fZ")
        else:
            lust_update = datetime.strptime(scenario_info["created_at"], "%Y-%m-%dT%H:%M:%S.%fZ")
        task_info_to_add = TaskInfoSchema(
            project_id=project_info["project_id"],
            base_scenario_id=project_info["base_scenario"]["id"],
            lust_update=lust_update
        )
        tasks[project_scenario_id] = TaskSchema(
            task_status=TaskStatusSchema(task_status="pending"),
            target_scenario_id=project_scenario_id,
            task_info=task_info_to_add
        )
        return {"action": "continue"}

def _evaluate_master_plan_task(project_scenario_id: int, token: str = Depends(auth.verify_token), is_context: bool = False):
    try:
        es.evaluate_f22(project_scenario_id, token, is_context=is_context)
        tasks[project_scenario_id].task_status.task_status = "success"
    except Exception as e:
        logger.error(e)
        logger.exception(e)
        tasks[project_scenario_id].task_status.task_status = 'error'
        tasks[project_scenario_id].task_status.error_info = e.__str__()


@router.post('/evaluate')
def evaluate(background_tasks: BackgroundTasks, project_scenario_id: int, token: str = Depends(auth.verify_token), is_context: bool = False):
    check_result = check_or_set_status(project_scenario_id, token)
    if check_result["action"] == "return":
        del check_result["action"]
        return check_result
    background_tasks.add_task(_evaluate_master_plan_task, project_scenario_id, token, is_context=is_context)
    return {'task_id': project_scenario_id}

#ручка для теста, можно убрать
@router.get('/get_scenario_zones/{scenario_id}')
async def get_scenario_zones(scenario_id: int):
    zones: gpd.GeoDataFrame = await get_scenario_functional_zones(scenario_id)
    geojson_str = zones.to_json()
    geojson_dict = json.loads(geojson_str)
    return JSONResponse(content=geojson_dict, media_type="application/geo+json")


# @router.post('/socio_economic_effects/{scenario_id}')
# def evaluate_socio_economic_effects(
#         scenario_id: int,
#         token: str = Depends(auth.verify_token),
#         functional_zone_source: str,
#         functional_zone_year: int,
#         context_functional_zone_source: str,
#         context_functional_zone_year: int
#
# ):
#

@router.get('/get_development_parameters/{scenario_id}')
async def get_development_parameters(scenario_id: int):
    development_parameters: gpd.GeoDataFrame = await run_development_parameters(scenario_id)
    geojson_str = development_parameters.to_json()
    geojson_dict = json.loads(geojson_str)
    return JSONResponse(content=geojson_dict, media_type="application/geo+json")



@router.delete('/evaluation')
def delete_evaluation(project_scenario_id : int):
    try:
        es.delete_evaluation(project_scenario_id)
        return 'oke'
    except:
        return 'oops'




