
import sqlite3
import json
from datetime import datetime, timezone

from part_d_retrieval import retrieve


# Tool 1: Get order details from SQLite
def get_order(order_id):
    connection = sqlite3.connect("data/instamart.db")
    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT order_id, status, order_amount, delivery_minutes
        FROM orders
        WHERE order_id = ?
        """,
        (order_id,)
    )

    row = cursor.fetchone()
    connection.close()

    if row is None:
        return {"error": "Order not found", "order_id": order_id}

    return {
        "order_id": row["order_id"],
        "status": row["status"],
        "order_amount": row["order_amount"],
        "delivery_minutes": row["delivery_minutes"]
    }


# Tool 2: Search policy documents
def search_policy(query):
    return {"results": retrieve(query, k=3)}


# Tool 3: Calculate refund amount
def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is None:
        return {"refund_inr": 0.0}

    if delivery_minutes > 45:
        return {"refund_inr": round(float(order_amount), 2)}

    return {"refund_inr": 0.0}


# Common function to write refund ledger
def write_ledger(tool_name, order_id, refund_inr):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": tool_name,
        "order_id": order_id,
        "refund_inr": refund_inr
    }

    with open(
        "outputs/refund_ledger.jsonl",
        "a",
        encoding="utf-8"
    ) as file:
        file.write(json.dumps(record) + "\n")

    return {
        "success": True,
        "order_id": order_id,
        "refund_inr": refund_inr
    }


# Tool 4: Issue refund
def issue_refund(order_id, refund_inr):
    return write_ledger(
        "issue_refund",
        order_id,
        refund_inr
    )


# Tool 5: Escalate to human
def escalate_to_human(order_id, refund_inr):
    return write_ledger(
        "escalate_to_human",
        order_id,
        refund_inr
    )


# Tool specifications
TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Retrieve order status, amount and delivery time.",
        "parameters": {
            "order_id": "string"
        }
    },
    {
        "name": "search_policy",
        "description": "Search policy documents for relevant information.",
        "parameters": {
            "query": "string"
        }
    },
    {
        "name": "calculate_refund",
        "description": "Calculate the refund for an eligible late delivery.",
        "parameters": {
            "order_amount": "number",
            "delivery_minutes": "number"
        }
    },
    {
        "name": "issue_refund",
        "description": "Record an approved refund in the refund ledger.",
        "parameters": {
            "order_id": "string",
            "refund_inr": "number"
        }
    },
    {
        "name": "escalate_to_human",
        "description": "Record a refund requiring human approval.",
        "parameters": {
            "order_id": "string",
            "refund_inr": "number"
        }
    }
]


# Map tool names to Python functions
TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}
