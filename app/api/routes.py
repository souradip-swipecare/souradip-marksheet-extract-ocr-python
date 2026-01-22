from fastapi import APIRouter
from app.api.v1 import routes
#  we add prefix under this router also but further we include  using sub route
api_router = APIRouter()

api_router.include_router(
    routes.router,
    prefix="/v1",
    tags=["extract router"]
)