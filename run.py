from __future__ import annotations

import argparse
import uuid

from dotenv import load_dotenv
from app.graphs.orchestrator import build_orchestrator_graph


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run Vitelis v0.2 pipeline")
    parser.add_argument("--company-name", required=True, help="Company name")
    parser.add_argument("--company-domain", required=True, help="Company domain (e.g., vodafone.com)")
    parser.add_argument("--run-id", default=str(uuid.uuid4())[:8], help="Optional run id")
    args = parser.parse_args()

    graph = build_orchestrator_graph()
    state = graph.invoke(
        {
            "run_id": args.run_id,
            "company_name": args.company_name,
            "company_domain": args.company_domain,
        }
    )

    print("Run complete")
    print(f"Report: {state.get('report_path')}")
    print(f"Overall score: {state.get('overall_score')}")


if __name__ == "__main__":
    main()
