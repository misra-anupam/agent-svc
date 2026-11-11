from fastapi import APIRouter

from agent_svc.api.v1 import agent_info

router = APIRouter(prefix="/v1")
router.include_router(agent_info.router)
