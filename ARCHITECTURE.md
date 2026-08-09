# Architecture Diagrams

## System Architecture Diagram (Implementation View)

```mermaid
flowchart TB
    subgraph Client["Client"]
        Browser["Browser"]
    end

    subgraph Frontend["business-intelligence-frontend (Next.js 16, React 19)"]
        Pages["App Router pages\n(landing, auth, dashboard/*)"]
        Lib["lib/api.ts, lib/auth.ts\nREST client + JWT session"]
        Query["TanStack React Query cache"]
    end

    subgraph Infra["EC2 host (Docker Compose)"]
        Nginx["nginx reverse proxy\n(deploy/nginx.conf)"]
        subgraph Backend["business-intelligence-backend (FastAPI, Uvicorn)"]
            MW["Middleware:\nAudit -> SecurityHeaders -> RateLimit -> CORS"]
            API["app/api/v1 routers\n(/api/v1/*)"]
            Services["app/services\n(analytics, etl, ml, ai, alerts, insights, reports)"]
            Scheduler["APScheduler (app/workers/scheduler.py)\nretrain, anomaly scan, insights/alerts, monthly report"]
            Core["app/core\nconfig, security (JWT), database"]
        end
    end

    subgraph External["External services"]
        Supabase[("Supabase\nPostgres + Auth (JWT issuer)")]
        S3[("AWS S3\nuploads / report files")]
        Groq["Groq LLM (primary)"]
        Gemini["Google Gemini (fallback)"]
        GoogleOAuth["Google OAuth"]
        SMTP["SMTP (alert emails)"]
    end

    Browser --> Pages
    Pages --> Lib --> Query
    Lib -->|HTTPS REST + JWT Bearer| Nginx
    Nginx --> MW --> API
    API --> Services
    API --> Core
    Services --> Core
    Scheduler --> Services
    Core -->|asyncpg| Supabase
    Core -->|verify JWT| Supabase
    Services -->|read/write files| S3
    Services -->|AI chat/insights| Groq
    Services -->|fallback| Gemini
    API -->|OAuth login| GoogleOAuth
    Services -->|alert channel| SMTP
```

## Use Case Diagram

```mermaid
flowchart LR
    Analyst((Analyst))
    Manager((Manager))
    Admin((Admin))

    subgraph System["BI & Decision Support Dashboard"]
        UC1["Log in / Sign up\n(email or Google OAuth)"]
        UC2["View KPI & analytics dashboards\n(sales, finance, inventory)"]
        UC3["View forecasts & anomalies"]
        UC4["Chat with AI assistant\n/ view AI insights"]
        UC5["View recommendations"]
        UC6["Receive & read notifications"]
        UC7["Manage data sources\n& upload files"]
        UC8["Trigger / monitor ETL jobs"]
        UC9["Define alert rules"]
        UC10["Generate & download reports"]
        UC11["Manage users & roles"]
        UC12["View audit logs"]
    end

    Analyst --> UC1
    Analyst --> UC2
    Analyst --> UC3
    Analyst --> UC4
    Analyst --> UC5
    Analyst --> UC6

    Manager --> UC1
    Manager --> UC2
    Manager --> UC3
    Manager --> UC4
    Manager --> UC5
    Manager --> UC6
    Manager --> UC7
    Manager --> UC8
    Manager --> UC9
    Manager --> UC10

    Admin --> UC1
    Admin --> UC2
    Admin --> UC7
    Admin --> UC8
    Admin --> UC9
    Admin --> UC10
    Admin --> UC11
    Admin --> UC12
```

## Sequence Diagram

Example flow: analyst uploads a data file, which is processed by ETL, feeds analytics/ML, and produces an insight surfaced back to the user.

```mermaid
sequenceDiagram
    actor User as Analyst (Browser)
    participant FE as Next.js Frontend
    participant API as FastAPI (/api/v1)
    participant Auth as Supabase Auth
    participant Store as Storage Service (S3/local)
    participant ETL as ETL Service
    participant DB as Postgres (Supabase)
    participant Sched as APScheduler
    participant AI as AI Provider (Groq/Gemini)

    User->>FE: Log in
    FE->>API: POST /auth/login
    API->>Auth: Verify credentials / issue JWT
    Auth-->>API: JWT
    API-->>FE: JWT + profile
    FE-->>User: Redirect to dashboard

    User->>FE: Upload sales file
    FE->>API: POST /uploads (multipart, Bearer JWT)
    API->>Store: Save raw file
    API->>DB: Insert raw_uploads row
    API-->>FE: 201 Created (upload id)

    User->>FE: Trigger ETL run
    FE->>API: POST /etl/run/{source_id}
    API->>ETL: Run ETL job
    ETL->>DB: Read raw_uploads, write sales_transactions/expenses/inventory_levels
    ETL->>DB: Update etl_jobs status = succeeded
    API-->>FE: Job status

    Sched->>ETL: Nightly insights/alerts job
    ETL->>DB: Read latest KPIs, forecasts, anomalies
    ETL->>AI: Request generated insight text
    AI-->>ETL: Insight narrative
    ETL->>DB: Insert insights, notifications rows

    User->>FE: Open dashboard
    FE->>API: GET /kpis/summary, /insights, /notifications
    API->>DB: Query aggregated data
    DB-->>API: Rows
    API-->>FE: JSON response
    FE-->>User: Render charts, insights, notifications
```

## Entity Relationship Diagram

```mermaid
erDiagram
    PROFILE ||--o{ AUDIT_LOG : "performs"
    PROFILE ||--o{ RAW_UPLOAD : "uploads"
    PROFILE ||--o{ ALERT_RULE : "creates"
    PROFILE ||--o{ NOTIFICATION : "receives"
    PROFILE ||--o{ REPORT : "generates"
    PROFILE ||--o{ ANOMALY : "acknowledges"
    PROFILE ||--o{ CONVERSATION : "starts"

    DATA_SOURCE ||--o{ ETL_JOB : "triggers"
    DATA_SOURCE ||--o{ RAW_UPLOAD : "receives"
    DATA_SOURCE ||--o{ SALES_TRANSACTION : "sources"
    DATA_SOURCE ||--o{ EXPENSE : "sources"
    DATA_SOURCE ||--o{ INVENTORY_LEVEL : "sources"

    PRODUCT ||--o{ SALES_TRANSACTION : "sold in"
    PRODUCT ||--o{ INVENTORY_LEVEL : "tracked in"
    CUSTOMER ||--o{ SALES_TRANSACTION : "makes"
    ETL_JOB ||--o{ SALES_TRANSACTION : "loads"
    ETL_JOB ||--o{ EXPENSE : "loads"

    ML_MODEL ||--o{ FORECAST : "produces"
    FORECAST ||--o{ INSIGHT : "informs"
    ANOMALY ||--o{ INSIGHT : "informs"
    ALERT_RULE ||--o{ NOTIFICATION : "fires"
    INSIGHT ||--o{ NOTIFICATION : "generates"

    CONVERSATION ||--o{ MESSAGE : "contains"

    PROFILE {
        uuid id PK
        string email
        string password_hash
        string full_name
        enum role
        string department
        bool is_active
        jsonb preferences
    }
    AUDIT_LOG {
        uuid id PK
        uuid user_id FK
        string action
        string entity
        string entity_id
        jsonb detail
        string ip_address
    }
    DATA_SOURCE {
        uuid id PK
        string name
        enum kind
        jsonb config
        enum target_domain
        string schedule_cron
        enum status
    }
    RAW_UPLOAD {
        uuid id PK
        uuid data_source_id FK
        uuid uploaded_by FK
        string file_name
        string s3_key
        int row_count
        string status
        jsonb error_report
    }
    ETL_JOB {
        uuid id PK
        uuid data_source_id FK
        enum trigger
        datetime started_at
        datetime finished_at
        enum status
        int rows_in
        int rows_loaded
        int rows_rejected
        jsonb log
    }
    PRODUCT {
        uuid id PK
        string sku
        string name
        string category
        decimal unit_cost
        decimal unit_price
    }
    CUSTOMER {
        uuid id PK
        string name
        enum segment
        string city
        string region
    }
    SALES_TRANSACTION {
        int id PK
        uuid product_id FK
        uuid customer_id FK
        uuid source_id FK
        uuid etl_job_id FK
        date txn_date
        int quantity
        decimal unit_price
        decimal discount
        decimal total_amount
        string channel
        string region
        string row_hash
    }
    EXPENSE {
        int id PK
        uuid source_id FK
        uuid etl_job_id FK
        date expense_date
        enum category
        decimal amount
        string department
        string description
        string row_hash
    }
    INVENTORY_LEVEL {
        int id PK
        uuid product_id FK
        uuid source_id FK
        date snapshot_date
        int quantity_on_hand
        int reorder_level
        string warehouse
    }
    KPI_SNAPSHOT {
        int id PK
        date snapshot_date
        string metric
        jsonb dimensions
        decimal value
    }
    ML_MODEL {
        uuid id PK
        enum model_type
        string target
        string version
        datetime trained_at
        int training_rows
        jsonb metrics
        jsonb params
        string artifact_s3_key
        bool is_active
    }
    FORECAST {
        int id PK
        uuid model_id FK
        string target
        date forecast_date
        int horizon_days
        decimal yhat
        decimal yhat_lower
        decimal yhat_upper
        datetime generated_at
    }
    ANOMALY {
        uuid id PK
        uuid acknowledged_by FK
        datetime detected_at
        string metric
        decimal observed_value
        decimal expected_value
        decimal deviation_score
        enum severity
        jsonb context
        enum status
    }
    INSIGHT {
        uuid id PK
        uuid related_anomaly_id FK
        uuid related_forecast_id FK
        enum insight_type
        string title
        text body
        enum severity
        jsonb evidence
        date period_start
        date period_end
        datetime generated_at
        bool is_pinned
        string dedupe_key
    }
    ALERT_RULE {
        uuid id PK
        uuid created_by FK
        string name
        string metric
        enum condition
        decimal threshold
        int window_days
        jsonb channels
        string_array roles_notified
        bool is_active
    }
    NOTIFICATION {
        uuid id PK
        uuid user_id FK
        uuid alert_rule_id FK
        uuid insight_id FK
        string title
        text body
        bool is_read
    }
    REPORT {
        uuid id PK
        uuid generated_by FK
        enum report_type
        date period_start
        date period_end
        enum format
        string s3_key
    }
    CONVERSATION {
        uuid id PK
        uuid user_id FK
        string title
    }
    MESSAGE {
        uuid id PK
        uuid conversation_id FK
        enum role
        text content
        jsonb metadata
    }
```
