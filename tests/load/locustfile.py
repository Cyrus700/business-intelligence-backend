"""Load test for the read-heavy dashboard path.

Run (no install needed):
    TOKEN=<jwt>  # e.g. NEXT_PUBLIC_DEV_API_TOKEN from the frontend .env.local
    uvx locust -f tests/load/locustfile.py --host http://localhost:8000 \
        --headless -u 25 -r 5 -t 60s
Acceptance (06-testing-strategy.md): p95 < 500 ms on dashboard reads at 25 users.
"""

import os

from locust import HttpUser, between, task

TOKEN = os.environ.get("TOKEN", "")
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


class DashboardUser(HttpUser):
    wait_time = between(1, 3)

    @task(5)
    def kpi_summary(self):
        self.client.get("/api/v1/kpis/summary", headers=HEADERS)

    @task(3)
    def revenue_timeseries(self):
        self.client.get("/api/v1/kpis/timeseries", params={"metric": "revenue"}, headers=HEADERS)

    @task(2)
    def sales_by_product(self):
        self.client.get("/api/v1/sales/by-product", headers=HEADERS)

    @task(2)
    def forecasts(self):
        self.client.get("/api/v1/forecasts", params={"horizon": 30}, headers=HEADERS)

    @task(1)
    def insights(self):
        self.client.get("/api/v1/insights", params={"limit": 10}, headers=HEADERS)

    @task(1)
    def anomalies(self):
        self.client.get("/api/v1/anomalies", params={"status": "open"}, headers=HEADERS)
