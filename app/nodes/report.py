from __future__ import annotations

from typing import Dict

from app.nodes.aggregate_report import aggregate_report_node


def report_node(state: Dict) -> Dict:
    return aggregate_report_node(state)
