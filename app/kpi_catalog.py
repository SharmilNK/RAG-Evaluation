from __future__ import annotations

from typing import List

import yaml

from app.models import KPIDefinition


def load_kpi_catalog(path: str | None = None) -> List[KPIDefinition]:
    if path is None:
        path = "app/kpis.yaml"
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or []
    return [KPIDefinition(**item) for item in data]
