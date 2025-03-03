import os
from datetime import datetime
from typing import Annotated

from loguru import logger
from blocksnet.models import ServiceType
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from ...utils import auth, const, decorators
from . import effects_models as em
from . import effects_service as es
from .services import service_type_service as sts
from app.api.routers.effects.task_schema import TaskSchema, TaskStatusSchema, TaskInfoSchema
from app.api.routers.effects.services.task_api_service import get_scenario_info, get_all_project_info, get_project_id

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

@router.get('/service_types')
def get_service_types(region_id: int) -> list[ServiceType]:
    return sts.get_bn_service_types(region_id)

@router.get('/provision_layer')
@decorators.gdf_to_geojson
def get_provision_layer(project_scenario_id: int, scale_type: em.ScaleType, service_type_id: int,
                        token: str = Depends(auth.verify_token)):
    check_task_evaluation(project_scenario_id)
    return es.get_provision_layer(project_scenario_id, scale_type, service_type_id, token)

@router.get('/provision_data')
def get_provision_data(
        project_scenario_id: int,
        scale_type: Annotated[em.ScaleTypeModel, Depends(em.ScaleTypeModel)],
        token: str = Depends(auth.verify_token)
):
    check_task_evaluation(project_scenario_id)
    return es.get_provision_data(project_scenario_id, scale_type.scale_type, token)

@router.get('/transport_layer')
@decorators.gdf_to_geojson
def get_transport_layer(project_scenario_id: int, scale_type: em.ScaleType, token: str = Depends(auth.verify_token)):
    check_task_evaluation(project_scenario_id)
    return es.get_transport_layer(project_scenario_id, scale_type, token)

@router.get('/transport_data')
def get_transport_data(project_scenario_id: int, scale_type: em.ScaleType, token: str = Depends(auth.verify_token)):
    check_task_evaluation(project_scenario_id)
    return es.get_transport_data(project_scenario_id, scale_type, token)

@router.get('/connectivity_layer')
@decorators.gdf_to_geojson
def get_connectivity_layer(project_scenario_id: int, scale_type: em.ScaleType, token: str = Depends(auth.verify_token)):
    check_task_evaluation(project_scenario_id)
    return es.get_connectivity_layer(project_scenario_id, scale_type, token)

@router.get('/connectivity_data')
def get_connectivity_data(project_scenario_id: int, scale_type: em.ScaleType, token: str = Depends(auth.verify_token)):
    check_task_evaluation(project_scenario_id)
    return es.get_connectivity_data(project_scenario_id, scale_type, token)

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

def _evaluate_effects_task(project_scenario_id: int, token: str):

    try:
        es.evaluate_effects(project_scenario_id, token)
        tasks[project_scenario_id].task_status.task_status = "success"
    except Exception as e:
        logger.error(e)
        logger.exception(e)
        tasks[project_scenario_id].task_status.task_status = 'error'
        tasks[project_scenario_id].task_status.error_info = e.__str__()

@router.post('/evaluate')
def evaluate(background_tasks: BackgroundTasks, project_scenario_id: int, token: str = Depends(auth.verify_token)):
    check_result = check_or_set_status(project_scenario_id, token)
    if check_result["action"] == "return":
        del check_result["action"]
        return check_result
    background_tasks.add_task(_evaluate_effects_task, project_scenario_id, token)
    return {'task_id' : project_scenario_id }

@router.delete('/evaluation')
def delete_evaluation(project_scenario_id : int):
    try:
        es.delete_evaluation(project_scenario_id)
        return 'oke'
    except:
        return 'oops'