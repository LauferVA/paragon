# API Design Template

## API Overview
**Purpose:** High-level description of the API.

**Questions if missing:**
- What is the purpose of this API?
- Who are the consumers (internal, external, public)?
- What API style is used (REST, GraphQL, gRPC, WebSocket)?
- What is the versioning strategy?

**Example:**
> A REST API for querying and managing service dependencies. Consumed by internal developer tools and CI/CD pipelines. Versioned using URL path (e.g., /v1/, /v2/).

---

## Authentication & Authorization
**Purpose:** Define how clients authenticate and what they can access.

**Questions if missing:**
- How do clients authenticate (API keys, OAuth, JWT)?
- What authorization model is used (RBAC, ABAC, ACL)?
- What are the permission scopes?
- How are credentials managed and rotated?
- What is the session/token lifetime?

**Example:**
> **Authentication:** Bearer tokens (JWT) obtained via OAuth 2.0
> **Authorization:** Role-based (RBAC)
> - `read:dependencies` - Read dependency data
> - `write:dependencies` - Update dependency data
> - `admin:system` - Administrative operations
>
> **Token lifetime:** 1 hour (refresh tokens valid for 30 days)

---

## Endpoints
**Purpose:** List all API endpoints with details.

**For each endpoint, specify:**
- Method (GET, POST, PUT, DELETE, etc.)
- Path
- Description
- Request parameters (path, query, body)
- Response schema
- Status codes
- Example request/response

**Questions if missing:**
- What are all the endpoints?
- What HTTP methods are used?
- What are the request/response formats?
- What are the possible status codes and error responses?

**Example:**

### GET /v1/services/{service_id}/dependencies
**Description:** Retrieve all dependencies for a service.

**Path Parameters:**
- `service_id` (string, required) - Unique service identifier

**Query Parameters:**
- `depth` (integer, optional, default=1) - How many levels deep to traverse
- `include_transitive` (boolean, optional, default=false) - Include transitive dependencies

**Response (200 OK):**
```json
{
  "service_id": "user-service",
  "dependencies": [
    {
      "service_id": "auth-service",
      "version": "2.3.1",
      "type": "DIRECT"
    }
  ]
}
```

**Error Responses:**
- `404 Not Found` - Service not found
- `401 Unauthorized` - Missing or invalid token

---

### POST /v1/services/{service_id}/analyze
**Description:** Trigger dependency analysis for a service.

**Path Parameters:**
- `service_id` (string, required)

**Request Body:**
```json
{
  "repo_url": "https://gitlab.com/org/service",
  "branch": "main",
  "force_refresh": false
}
```

**Response (202 Accepted):**
```json
{
  "job_id": "abc-123",
  "status": "queued",
  "estimated_duration_seconds": 120
}
```

---

## Data Models
**Purpose:** Define the schemas for request/response bodies.

**Questions if missing:**
- What are the data models/schemas?
- What are the field types and constraints?
- What fields are required vs optional?
- What are the validation rules?

**Example:**

### ServiceDependency
```json
{
  "service_id": "string (required, 1-100 chars)",
  "name": "string (required)",
  "version": "string (semver format)",
  "type": "enum: DIRECT | TRANSITIVE",
  "repo_url": "string (URL format)",
  "last_updated": "string (ISO 8601 timestamp)"
}
```

### Conflict
```json
{
  "conflict_id": "string (required)",
  "services": ["string", "string"],
  "reason": "string (required)",
  "severity": "enum: HIGH | MEDIUM | LOW",
  "suggested_resolution": "string (optional)"
}
```

---

## Error Handling
**Purpose:** Define error response format and codes.

**Questions if missing:**
- What is the error response schema?
- What error codes are used?
- How are errors categorized?
- What information is included in error messages?

**Example:**

### Error Response Format
```json
{
  "error": {
    "code": "INVALID_SERVICE_ID",
    "message": "Service ID must be 1-100 characters",
    "details": {
      "field": "service_id",
      "value": "",
      "constraint": "length(1, 100)"
    },
    "request_id": "req-abc-123"
  }
}
```

### Error Codes
- `INVALID_REQUEST` - Malformed request
- `UNAUTHORIZED` - Authentication failure
- `FORBIDDEN` - Authorization failure
- `NOT_FOUND` - Resource not found
- `CONFLICT` - Resource conflict
- `RATE_LIMIT_EXCEEDED` - Too many requests
- `INTERNAL_ERROR` - Server error

---

## Rate Limiting
**Purpose:** Define rate limits to prevent abuse.

**Questions if missing:**
- What are the rate limits?
- How are limits enforced (per user, per IP, per API key)?
- What happens when limits are exceeded?
- How can clients check their usage?

**Example:**
> **Limits:**
> - 100 requests per minute per API key
> - 1,000 requests per hour per API key
>
> **Enforcement:** Token bucket algorithm
>
> **Response when exceeded (429 Too Many Requests):**
> ```json
> {
>   "error": {
>     "code": "RATE_LIMIT_EXCEEDED",
>     "message": "Rate limit exceeded",
>     "retry_after_seconds": 42
>   }
> }
> ```
>
> **Headers:**
> - `X-RateLimit-Limit: 100`
> - `X-RateLimit-Remaining: 0`
> - `X-RateLimit-Reset: 1638360000`

---

## Pagination
**Purpose:** Define how large result sets are paginated.

**Questions if missing:**
- How are large result sets paginated?
- What pagination style is used (offset, cursor)?
- What are the default and max page sizes?
- How do clients navigate pages?

**Example:**
> **Style:** Cursor-based pagination
>
> **Request:**
> ```
> GET /v1/services?limit=50&cursor=eyJpZCI6MTIzfQ
> ```
>
> **Response:**
> ```json
> {
>   "data": [...],
>   "pagination": {
>     "next_cursor": "eyJpZCI6MTczfQ",
>     "has_more": true,
>     "total_count": 500
>   }
> }
> ```
>
> **Limits:**
> - Default: 50 items
> - Max: 200 items

---

## Versioning
**Purpose:** Define how API versions are managed.

**Questions if missing:**
- How are API versions specified (URL, header)?
- What is the deprecation policy?
- How long are old versions supported?
- How are breaking changes communicated?

**Example:**
> **Strategy:** URL path versioning (`/v1/`, `/v2/`)
>
> **Deprecation:**
> - 6 months notice before deprecation
> - Deprecated endpoints return `X-API-Deprecated: true` header
> - Sunset date in `Sunset` header (RFC 8594)
>
> **Breaking Changes:**
> - Announced in changelog and developer emails
> - Migration guide provided

---

## WebSockets / Real-time
**Purpose:** Define real-time communication if applicable.

**Questions if missing:**
- Are there real-time features (WebSockets, SSE)?
- What events are pushed to clients?
- What is the connection lifecycle?
- How is authentication handled for persistent connections?

**Example:**
> **Protocol:** WebSocket
>
> **Endpoint:** `wss://api.example.com/v1/stream`
>
> **Authentication:** JWT in query parameter `?token=<jwt>`
>
> **Message Format:**
> ```json
> {
>   "type": "DEPENDENCY_UPDATED",
>   "service_id": "user-service",
>   "timestamp": "2025-12-07T10:00:00Z",
>   "data": {...}
> }
> ```
>
> **Event Types:**
> - `DEPENDENCY_UPDATED` - Dependency changed
> - `CONFLICT_DETECTED` - New conflict found
> - `ANALYSIS_COMPLETE` - Analysis job finished

---

## Performance & Caching
**Purpose:** Define caching strategies and performance expectations.

**Questions if missing:**
- What are the performance SLAs (latency, throughput)?
- What caching strategies are used?
- What are the cache invalidation rules?
- What HTTP caching headers are used?

**Example:**
> **SLAs:**
> - P50 latency: < 100ms
> - P99 latency: < 500ms
> - Availability: 99.9%
>
> **Caching:**
> - CDN caching for static endpoints (1 hour)
> - Application-level caching for expensive queries (5 minutes)
>
> **Headers:**
> - `Cache-Control: public, max-age=3600` (for cacheable responses)
> - `ETag` for conditional requests
> - `If-None-Match` supported (304 Not Modified)

---

## SDKs & Client Libraries
**Purpose:** List available SDKs and usage examples.

**Questions if missing:**
- Are there official SDKs/client libraries?
- What languages are supported?
- Where is the SDK documentation?
- How are SDKs versioned?

**Example:**
> **Official SDKs:**
> - Python: `pip install paragon-deps-sdk`
> - TypeScript: `npm install @paragon/deps-sdk`
> - Go: `go get github.com/paragon/deps-sdk-go`
>
> **Usage Example (Python):**
> ```python
> from paragon_deps import Client
>
> client = Client(api_key="...")
> deps = client.get_dependencies("user-service")
> ```

---

## Open Questions
**Purpose:** Track unresolved API design decisions.

**Questions if missing:**
- What API design decisions are pending?
- What trade-offs are being considered?

**Example:**
> - Should we support GraphQL in addition to REST?
> - Should we use HTTP/2 server push for real-time updates?
> - What is the right default page size (50 vs 100)?
