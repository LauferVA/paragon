# API Service Domain Template

This template extends the generic templates with API/backend service-specific questions.

## API Architecture Pattern
**Questions to ask:**
- What API architecture will you use (REST, GraphQL, gRPC, WebSocket)?
- Will this be a monolithic API or microservice?
- What is the service boundary (what does this API own)?
- How does this fit into the larger system architecture?
- Will you use an API gateway or direct service-to-service calls?

**Example answers:**
> - REST API with JSON
> - Microservice for dependency management
> - Owns: Service metadata, dependency graph, conflict detection
> - Fits in: Part of developer tooling ecosystem
> - API gateway: Kong for routing + rate limiting

---

## Request/Response Handling
**Questions to ask:**
- What request formats are supported (JSON, XML, protobuf)?
- What response formats are supported?
- Will you use HTTP/1.1, HTTP/2, or HTTP/3?
- How will you handle content negotiation (Accept headers)?
- What character encoding is used (UTF-8)?
- Will you support request/response compression (gzip, br)?

**Example answers:**
> - Request: JSON only (Content-Type: application/json)
> - Response: JSON only
> - HTTP/2 for improved performance
> - Content negotiation: Accept JSON or return 406
> - Encoding: UTF-8 everywhere
> - Compression: gzip for responses > 1KB

---

## Database & Persistence
**Questions to ask:**
- What database will you use (PostgreSQL, MongoDB, SQLite, etc.)?
- What ORM or database library will you use (SQLAlchemy, Prisma, raw SQL)?
- How will you handle database migrations?
- What is the database schema?
- How will you handle database connections (pooling, timeouts)?
- Do you need database replication or sharding?

**Example answers:**
> - Database: SQLite for embedded graph storage
> - Library: Direct SQL via aiosqlite (async)
> - Migrations: Alembic for schema versioning
> - Schema: Services table, dependencies table, metadata JSONB
> - Connections: Single connection (SQLite limitation)
> - Replication: Not needed for v1 (single-instance)

---

## Data Validation & Serialization
**Questions to ask:**
- How will you validate incoming requests (schema, types)?
- What validation library will you use (Pydantic, marshmallow, Zod)?
- How will you serialize/deserialize data (msgspec, JSON, protobuf)?
- How will you handle validation errors (400 responses)?
- Will you use strict or lenient validation?

**Example answers:**
> - Validation: msgspec.Struct with strict validation
> - Library: msgspec for performance (no Pydantic per project rules)
> - Serialization: msgspec JSON encoder/decoder
> - Validation errors: 400 Bad Request with detailed field errors
> - Mode: Strict (reject unknown fields)

---

## Authentication & Authorization
**Questions to ask:**
- How will clients authenticate (API keys, JWT, OAuth)?
- How will you authorize requests (RBAC, ABAC, scopes)?
- Where are credentials verified (local, external auth service)?
- How will you handle token expiry and refresh?
- Will you support service-to-service authentication (mutual TLS)?
- How will you rate limit per user/API key?

**Example answers:**
> - Auth: JWT tokens from central auth service
> - Authz: RBAC with roles (admin, developer, read-only)
> - Verification: Validate JWT signature using public key
> - Expiry: Return 401, client must refresh
> - Service-to-service: mTLS for internal calls
> - Rate limit: Per API key (100 req/min)

---

## Error Handling
**Questions to ask:**
- What error response format will you use?
- What HTTP status codes will you return?
- How will you categorize errors (validation, auth, server, etc.)?
- Will you include error codes or just messages?
- How will you handle unexpected exceptions?
- Will you expose stack traces (dev vs prod)?

**Example answers:**
> - Format: RFC 7807 Problem Details (application/problem+json)
> - Status codes: 200, 201, 400, 401, 403, 404, 409, 429, 500, 503
> - Categories: CLIENT_ERROR, AUTH_ERROR, SERVER_ERROR, RATE_LIMIT
> - Error codes: Yes (e.g., "INVALID_SERVICE_ID")
> - Exceptions: Catch all, return 500 with generic message
> - Stack traces: Only in dev/staging, never in production

---

## Rate Limiting & Throttling
**Questions to ask:**
- Will you implement rate limiting?
- What rate limit algorithm (token bucket, leaky bucket, fixed window)?
- What are the limits (per second, per minute, per hour)?
- How are limits enforced (per IP, per API key, per user)?
- What headers will you return (X-RateLimit-*)?
- How will clients know when to retry (Retry-After)?

**Example answers:**
> - Rate limiting: Yes, essential for public API
> - Algorithm: Token bucket (allows bursts)
> - Limits: 100 req/min, 1000 req/hour per API key
> - Enforcement: Per API key (identified by auth token)
> - Headers: X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset
> - Retry: Retry-After header (seconds until reset)

---

## Caching Strategy
**Questions to ask:**
- Will you use caching (in-memory, Redis, CDN)?
- What will be cached (responses, database queries, computed results)?
- What is the cache invalidation strategy (TTL, event-based)?
- How will you handle cache consistency?
- Will you use HTTP caching headers (ETag, Cache-Control)?
- Will you support conditional requests (If-None-Match)?

**Example answers:**
> - Caching: In-memory LRU cache + Redis for distributed
> - Cached: Service metadata (5min TTL), dependency queries (1min TTL)
> - Invalidation: TTL + explicit invalidation on writes
> - Consistency: Write-through cache (update on write)
> - HTTP caching: Cache-Control: public, max-age=300
> - Conditional: Support ETag + If-None-Match (304 Not Modified)

---

## Async & Concurrency
**Questions to ask:**
- Will you use async/await or blocking I/O?
- What concurrency model (threads, processes, async event loop)?
- How many concurrent requests can you handle?
- How will you handle long-running operations (background jobs)?
- Will you use a task queue (Celery, RQ, custom)?
- How will you prevent resource exhaustion?

**Example answers:**
> - Async: Yes, using asyncio throughout
> - Model: Async event loop (uvloop for performance)
> - Concurrency: 1000+ concurrent connections per worker
> - Long operations: Background jobs using asyncio.create_task()
> - Task queue: Redis-backed queue for heavy analysis jobs
> - Protection: Semaphore limiting concurrent DB connections

---

## Observability & Monitoring
**Questions to ask:**
- What metrics will you track (latency, throughput, errors)?
- What monitoring system will you use (Prometheus, Datadog, CloudWatch)?
- What log format will you use (JSON, plaintext)?
- How will you implement distributed tracing (OpenTelemetry, Jaeger)?
- Will you track custom business metrics?
- What alerts will you configure?

**Example answers:**
> - Metrics: Request latency (p50/p95/p99), error rate, active connections
> - System: Prometheus for metrics, Grafana for dashboards
> - Logs: JSON structured logs to stdout (captured by infra)
> - Tracing: OpenTelemetry with Jaeger backend
> - Business metrics: Services analyzed per hour, conflict detection rate
> - Alerts: Error rate > 5%, p99 latency > 1s, no heartbeat for 5min

---

## Health Checks & Readiness
**Questions to ask:**
- What health check endpoints will you expose?
- What checks will be performed (database, external services)?
- How will you distinguish liveness vs readiness?
- What response format for health checks?
- How often will health checks be called?

**Example answers:**
> - Endpoints: `/health` (liveness), `/ready` (readiness)
> - Liveness: Process is running (always 200 unless crashed)
> - Readiness: Database accessible, cache available
> - Format: JSON with component status or simple 200/503
> - Frequency: Every 10 seconds by Kubernetes

---

## Deployment & Scaling
**Questions to ask:**
- How will the service be deployed (Docker, K8s, serverless)?
- How will you handle zero-downtime deployments?
- What is the scaling strategy (horizontal, vertical)?
- How will you handle stateful operations during scaling?
- What resource limits (CPU, memory)?
- Will you use autoscaling?

**Example answers:**
> - Deployment: Docker containers on Kubernetes
> - Zero-downtime: Rolling updates with readiness checks
> - Scaling: Horizontal (add more pods)
> - Stateful: Use Redis for shared state across instances
> - Limits: 1 CPU, 2GB RAM per pod
> - Autoscaling: HPA based on CPU (50% target) + custom metrics (request queue)

---

## API Versioning
**Questions to ask:**
- How will you version the API (URL, header, query param)?
- What is the deprecation policy?
- How many versions will you support concurrently?
- How will you communicate breaking changes?
- Will you use semantic versioning?

**Example answers:**
> - Versioning: URL path (/v1/, /v2/)
> - Deprecation: 6 months notice, then 6 months sunset
> - Concurrent: Current + previous major version
> - Communication: Changelog, email to API consumers, X-API-Deprecated header
> - SemVer: Yes, v1.x.y for backward-compatible changes

---

## Data Consistency & Transactions
**Questions to ask:**
- What consistency guarantees do you need (strong, eventual)?
- Will you use database transactions?
- How will you handle distributed transactions (saga, 2PC)?
- What is your approach to concurrent writes (optimistic/pessimistic locking)?
- How will you handle race conditions?

**Example answers:**
> - Consistency: Strong consistency for critical operations
> - Transactions: Yes, ACID transactions via SQLite
> - Distributed: Not applicable (single database)
> - Concurrent writes: Optimistic locking with version field
> - Race conditions: Use database unique constraints + retry logic

---

## Security Hardening
**Questions to ask:**
- How will you prevent SQL injection (parameterized queries)?
- How will you prevent XSS (if serving HTML/JSON consumed by web)?
- How will you handle CORS (if browser-accessible)?
- Will you use HTTPS/TLS (certificate management)?
- How will you validate input to prevent injection attacks?
- Will you implement security headers (CSP, HSTS, etc.)?

**Example answers:**
> - SQL injection: Parameterized queries only, never string interpolation
> - XSS: JSON-only API (no HTML), but escape strings in error messages
> - CORS: Restrictive policy, whitelist allowed origins
> - TLS: Yes, required (redirect HTTP to HTTPS), cert from Let's Encrypt
> - Input validation: Strict schema validation, length limits, allowlists
> - Headers: HSTS, X-Content-Type-Options, X-Frame-Options

---

## API Documentation
**Questions to ask:**
- How will you document the API (OpenAPI, GraphQL schema)?
- Will you generate docs automatically or write manually?
- What documentation format (Swagger UI, Redoc, custom)?
- Will you provide code examples in multiple languages?
- How will you keep docs in sync with implementation?

**Example answers:**
> - Format: OpenAPI 3.0 specification
> - Generation: Auto-generated from code using FastAPI/apispec
> - UI: Swagger UI at /docs, Redoc at /redoc
> - Examples: Yes, curl + Python + TypeScript examples
> - Sync: Docs generated from code (single source of truth)

---

## Testing Strategy
**Questions to ask:**
- What types of tests (unit, integration, contract, load)?
- How will you test API endpoints (TestClient, real HTTP)?
- How will you test authentication/authorization?
- Will you use contract testing (Pact) for API consumers?
- How will you test error scenarios?
- What is your load testing strategy?

**Example answers:**
> - Unit: Individual functions, business logic
> - Integration: Full request/response with test database
> - Contract: OpenAPI schema validation
> - Auth: Mock JWT tokens in tests
> - Errors: Test all error paths (400, 401, 500)
> - Load: Locust tests simulating 1000 concurrent users

---

## Graceful Shutdown
**Questions to ask:**
- How will you handle graceful shutdown (SIGTERM)?
- Will you drain in-flight requests?
- What is the shutdown timeout?
- How will you notify clients during shutdown?

**Example answers:**
> - Shutdown: Listen for SIGTERM, stop accepting new requests
> - Drain: Wait for in-flight requests (max 30s)
> - Timeout: Force shutdown after 30s
> - Notification: Return 503 Service Unavailable for new requests
