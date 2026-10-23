import json
from pathlib import Path

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
ORDERS = json.loads((ROOT / "data" / "orders.json").read_text())["orders"]


class LookupArgs(BaseModel):
    order_id: str = Field(description="The order id, such as A-1001")


async def lookup_order(args: dict) -> dict:
    order_id = args["order_id"].strip().upper()
    return ORDERS.get(order_id, {"error": f"there is no order {order_id}"})
