# Data Pipeline Domain Template

This template extends the generic templates with data pipeline-specific questions.

## Pipeline Architecture
**Questions to ask:**
- What type of pipeline (batch, streaming, hybrid)?
- What orchestration tool (Airflow, Prefect, Dagster, custom)?
- What is the trigger mechanism (schedule, event, manual)?
- What is the data flow (source → transform → sink)?
- How many stages/steps are in the pipeline?
- What are the dependencies between pipeline stages?

**Example answers:**
> - Type: Batch pipeline with daily schedule
> - Orchestration: Apache Airflow (existing infrastructure)
> - Trigger: Daily at 2 AM UTC + manual trigger for backfills
> - Flow: GitLab API → Parse dependencies → Build graph → Store SQLite
> - Stages: Extract (1h), Transform (30m), Load (10m)
> - Dependencies: Sequential stages, parallel extraction per service

---

## Data Sources
**Questions to ask:**
- What are the data sources (databases, APIs, files, streams)?
- What data formats (JSON, CSV, Parquet, Avro)?
- How will you authenticate to sources?
- What is the data volume (MB, GB, TB)?
- What is the data velocity (records/second)?
- Are sources reliable (need retry logic)?
- How will you handle source schema changes?

**Example answers:**
> - Sources: GitLab API (REST), GitHub API (GraphQL)
> - Format: JSON responses
> - Auth: Personal access tokens (stored in secret manager)
> - Volume: ~10GB per day (500 services × 20MB each)
> - Velocity: Batch (not real-time), ~100 requests/minute
> - Reliability: APIs can fail, implement exponential backoff retry
> - Schema changes: Version API requests, validate response schemas

---

## Data Extraction
**Questions to ask:**
- Will you do full extraction or incremental?
- How will you track what has been extracted (watermarks, timestamps)?
- How will you handle pagination for large datasets?
- Will you parallelize extraction?
- How will you handle API rate limits?
- What happens if extraction fails mid-way?

**Example answers:**
> - Mode: Incremental (only changed services since last run)
> - Tracking: Store last_updated timestamp per service in metadata DB
> - Pagination: Handle GitLab pagination (100 items per page)
> - Parallelization: asyncio to fetch 10 services concurrently
> - Rate limits: Respect X-RateLimit headers, implement backoff
> - Failure: Checkpoint progress every 50 services, resume from checkpoint

---

## Data Transformation
**Questions to ask:**
- What transformations are needed (clean, enrich, aggregate, join)?
- Will transformations be stateless or stateful?
- What is the transformation logic (Python, SQL, Spark)?
- How will you handle invalid or missing data?
- Will you need to join data from multiple sources?
- How will you ensure transformation idempotency?

**Example answers:**
> - Transformations:
>   1. Parse source code to extract dependencies
>   2. Resolve version constraints
>   3. Build dependency graph
>   4. Detect conflicts
> - State: Stateful (need to accumulate dependencies for graph)
> - Logic: Python + tree-sitter for parsing, rustworkx for graph
> - Invalid data: Log warnings, skip invalid services, continue processing
> - Joins: Join service metadata with dependency data
> - Idempotency: Use deterministic IDs, upsert instead of insert

---

## Data Quality & Validation
**Questions to ask:**
- What data quality checks will you perform?
- How will you validate data schemas?
- What are the acceptable error thresholds?
- How will you handle data anomalies (outliers, duplicates)?
- Will you use data profiling or monitoring?
- What happens when data quality checks fail?

**Example answers:**
> - Quality checks:
>   - Schema validation (msgspec)
>   - Completeness (no null required fields)
>   - Uniqueness (no duplicate service IDs)
>   - Consistency (dependency targets must exist)
> - Thresholds: Fail if > 5% of services fail validation
> - Anomalies: Flag for manual review, continue pipeline
> - Monitoring: Track validation metrics over time (Prometheus)
> - Failure: Halt pipeline, send alert, require manual intervention

---

## Data Storage & Sinks
**Questions to ask:**
- Where will processed data be stored (database, data lake, warehouse)?
- What storage format (tables, files, objects)?
- Will you partition data (by date, by entity)?
- How will you handle schema evolution?
- Will you keep historical data or only latest?
- What is the data retention policy?

**Example answers:**
> - Storage: SQLite database + S3 for backups
> - Format: Normalized tables (services, dependencies, conflicts)
> - Partitioning: Not needed (single database per environment)
> - Schema evolution: Use database migrations (Alembic)
> - History: Keep all historical dependency changes (audit trail)
> - Retention: Keep data for 1 year, then archive to cold storage

---

## Error Handling & Recovery
**Questions to ask:**
- How will you handle transient errors (retry logic)?
- How will you handle permanent errors (dead-letter queue)?
- Will you support pipeline retries (full or partial)?
- How will you rollback failed pipelines?
- How will you alert on failures?
- What is the recovery time objective (RTO)?

**Example answers:**
> - Transient: Retry with exponential backoff (max 3 attempts)
> - Permanent: Log to error table, send alert, continue with other services
> - Retries: Support full pipeline retry or single-stage retry
> - Rollback: Restore from previous SQLite backup (< 1 hour old)
> - Alerts: PagerDuty for critical failures, Slack for warnings
> - RTO: Restore pipeline within 4 hours

---

## Scheduling & Triggers
**Questions to ask:**
- What is the pipeline schedule (cron, interval)?
- Will you support event-driven triggers?
- What is the expected pipeline duration?
- How will you handle overlapping runs?
- Will you support backfills for historical data?
- How will you handle holidays or maintenance windows?

**Example answers:**
> - Schedule: Daily at 2 AM UTC (low-traffic period)
> - Events: Webhook from GitLab on code push (optional real-time mode)
> - Duration: 90 minutes (1h extract, 30m transform, 10m load)
> - Overlapping: Prevent (use Airflow pool with max_active_runs=1)
> - Backfills: Support date range backfills via manual trigger
> - Maintenance: Skip runs on Sunday (use calendar in Airflow)

---

## Monitoring & Observability
**Questions to ask:**
- What metrics will you track (throughput, latency, errors)?
- How will you monitor pipeline health?
- Will you track data lineage?
- How will you visualize pipeline status (dashboards)?
- What alerts will you configure?
- How will you debug failed pipelines?

**Example answers:**
> - Metrics:
>   - Records processed per stage
>   - Pipeline duration per stage
>   - Error rate
>   - Data quality score
> - Health: Airflow UI + Prometheus + Grafana
> - Lineage: Track source → intermediate → final tables
> - Visualization: Grafana dashboard with pipeline SLAs
> - Alerts: Pipeline failure, duration > SLA, data quality < threshold
> - Debugging: Structured logs with correlation IDs, rerun single tasks

---

## Performance & Scalability
**Questions to ask:**
- What is the expected data volume growth?
- How will you scale the pipeline (vertical, horizontal)?
- What are the performance bottlenecks?
- Will you use parallel processing (Spark, Dask, multiprocessing)?
- How will you optimize resource usage (CPU, memory, I/O)?
- What are the performance SLAs?

**Example answers:**
> - Growth: 20% per year (100 services → 500 services in 3 years)
> - Scaling: Horizontal (Airflow workers), increase parallelism
> - Bottlenecks: API rate limits, code parsing (CPU-bound)
> - Parallelism: asyncio for I/O, multiprocessing for CPU-bound parsing
> - Optimization: LRU cache for parsed ASTs, batch API requests
> - SLA: Complete within 2 hours (buffer for delays)

---

## State Management
**Questions to ask:**
- Does the pipeline need to maintain state between runs?
- How will you store state (database, files, distributed cache)?
- How will you handle state corruption or loss?
- Will you use checkpointing for long-running pipelines?
- How will you ensure state consistency?

**Example answers:**
> - State: Yes, track last_extracted timestamp per service
> - Storage: Metadata table in SQLite (service_id, last_extracted_at)
> - Corruption: Validate state on pipeline start, reset if invalid
> - Checkpointing: Checkpoint every 50 services (resume on failure)
> - Consistency: Use database transactions for state updates

---

## Data Privacy & Compliance
**Questions to ask:**
- Does the pipeline process PII or sensitive data?
- What compliance requirements apply (GDPR, HIPAA, SOC2)?
- How will you anonymize or mask sensitive data?
- What is the data access control model?
- How will you audit data access?
- What is the data encryption strategy (at-rest, in-transit)?

**Example answers:**
> - PII: No PII in service metadata (only technical data)
> - Compliance: SOC2 (audit logging required)
> - Masking: Not needed (no sensitive data)
> - Access: Role-based, only authorized services can write
> - Audit: Log all pipeline runs with user, timestamp, data volumes
> - Encryption: TLS for GitLab API, SQLite encryption at rest

---

## Testing Strategy
**Questions to ask:**
- How will you test the pipeline (unit, integration, end-to-end)?
- Will you use test fixtures or synthetic data?
- How will you test data transformations?
- How will you test error scenarios?
- How will you test performance at scale?
- Will you use data regression testing?

**Example answers:**
> - Unit: Test individual transformation functions
> - Integration: Run pipeline on sample data (10 services)
> - E2E: Full pipeline on staging environment
> - Fixtures: Mock GitLab API responses for deterministic tests
> - Transformations: Property-based tests (Hypothesis) for parsing
> - Errors: Inject failures (API timeout, invalid JSON)
> - Performance: Test with 1000 services, ensure < 2h completion
> - Regression: Compare output against known-good baseline

---

## Dependency Management
**Questions to ask:**
- What external services does the pipeline depend on?
- How will you handle dependency failures?
- Will you use service discovery or hard-coded URLs?
- How will you handle dependency version changes?
- Will you use circuit breakers for unreliable dependencies?

**Example answers:**
> - Dependencies: GitLab API, Redis (optional cache)
> - Failures: Retry GitLab API, degrade gracefully without Redis
> - Discovery: Environment variables for URLs (12-factor config)
> - Versions: Pin GitLab API version (v4), monitor deprecations
> - Circuit breakers: Yes, open circuit after 5 consecutive API failures

---

## Cost Optimization
**Questions to ask:**
- What are the cost drivers (compute, storage, network, API calls)?
- How will you optimize costs?
- Will you use spot instances or reserved capacity?
- How will you monitor and alert on cost overruns?
- What is the budget for pipeline operations?

**Example answers:**
> - Costs: GitLab API calls (free tier), compute (AWS EC2), storage (S3)
> - Optimization:
>   - Use incremental extraction (reduce API calls)
>   - Cache parsed results (reduce compute)
>   - Compress S3 backups (reduce storage)
> - Compute: Use spot instances for non-critical pipelines
> - Monitoring: CloudWatch cost alerts (> $100/month)
> - Budget: $50/month target

---

## Disaster Recovery
**Questions to ask:**
- What is the backup strategy for pipeline data and state?
- How will you restore from backup?
- What is the recovery point objective (RPO)?
- How will you test disaster recovery procedures?
- What happens if the pipeline is down for extended periods?

**Example answers:**
> - Backup: Daily SQLite backup to S3 (retained for 30 days)
> - Restore: Download backup from S3, replace current DB
> - RPO: 24 hours (can lose up to 1 day of data)
> - Testing: Quarterly DR drill (restore from backup, verify data)
> - Extended downtime: Backfill missed days once pipeline is restored

---

## Pipeline Versioning & Deployment
**Questions to ask:**
- How will you version the pipeline code?
- How will you deploy pipeline changes (CI/CD)?
- Will you use blue/green or canary deployments?
- How will you rollback failed deployments?
- How will you handle breaking changes to data schemas?

**Example answers:**
> - Versioning: Git tags for releases (v1.0.0, v1.1.0)
> - Deployment: GitLab CI/CD pipeline
>   1. Run tests
>   2. Deploy to staging
>   3. Run smoke tests
>   4. Deploy to production
> - Strategy: Blue/green (run new version in parallel, switch over)
> - Rollback: Revert Airflow DAG to previous version
> - Breaking changes: Deploy schema migration first, then code
