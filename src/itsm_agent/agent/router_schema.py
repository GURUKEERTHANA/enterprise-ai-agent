# src/itsm_agent/agent/router_schema.py
from typing import Literal
from pydantic import BaseModel, Field

DepartmentLiteral = Literal[
    "IT Software",
    "DT-GPS",
    "Global People Live Chat Agents",
    "WPS - Badging",
    "Other",
]


class RouteAction(BaseModel):
    source_type: Literal["kb_article", "incident", "out_of_scope"] = Field(
        ...,
        description="Must be 'kb_article' for how-to, 'incident' for problems, or 'out_of_scope' for non-work queries.",
    )
    department_id: DepartmentLiteral = Field(
        ...,
        description="You MUST pick the best department. For Okta/VPN/Login, you MUST pick 'DT-GPS'.",
    )
    refined_query: str = Field(..., description="The search query.")
