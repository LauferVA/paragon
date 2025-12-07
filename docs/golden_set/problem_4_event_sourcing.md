# GOLDEN SET PROBLEM 4: Event Sourcing Message Queue

**Problem ID:** `GOLDEN-004`
**Category:** Complex Orchestration - Distributed Systems
**Difficulty:** High
**Estimated Implementation Time:** 4-6 days
**Date Created:** 2025-12-07

---

## EXECUTIVE SUMMARY

Build an event sourcing system with a persistent message queue, event replay, snapshotting, and eventual consistency guarantees. This problem tests the orchestrator's ability to:
- Design append-only, immutable data structures
- Implement message ordering and delivery guarantees
- Handle distributed system failure scenarios (replay, recovery)
- Manage consumer groups with load balancing
- Implement snapshot optimization for fast state recovery
- Ensure eventual consistency across event streams

This is a canonical distributed systems problem requiring careful state management, concurrency control, and failure recovery mechanisms.

---

## 1. PROBLEM STATEMENT

### 1.1 User Requirement (What a User Would Submit)

```
I need an event sourcing system that:
1. Stores all state changes as an append-only log of events
2. Supports multiple consumers reading events in order
3. Can replay events to rebuild application state from scratch
4. Creates snapshots to speed up recovery (don't replay from beginning)
5. Guarantees events are processed at least once (no lost events)
6. Handles consumer failures gracefully (dead letter queue)
7. Supports consumer groups for load balancing
8. Provides strong ordering guarantees within a partition

USE CASE:
I'm building a distributed order management system where:
- Each order is a stream of events: OrderCreated, ItemAdded, OrderSubmitted, PaymentProcessed, OrderShipped
- Multiple services consume events (inventory, billing, shipping)
- System must recover from crashes without data loss
- Audit trail must be immutable and complete
```

### 1.2 Success Criteria

**Functional Requirements:**
- Event log is append-only and immutable
- Events are persisted durably (survive process crashes)
- Consumers receive events in order (per partition)
- Event replay reconstructs exact state
- Snapshots reduce replay time by 90%+
- Dead letter queue captures failed events
- Consumer groups balance load across instances

**Quality Requirements:**
- All graph invariants maintained
- Test coverage ≥ 90%
- Event ordering: 100% correct (no reordering)
- Durability: 100% (no data loss)
- Performance: 10,000 events/sec ingestion, 1ms p99 latency
- Recovery time: <5 seconds for 1M event replay with snapshots

---

## 2. CORE COMPONENTS TO IMPLEMENT

### 2.1 Component Breakdown

#### Component 1: Event Store
**Type:** Core persistence layer
**File Path:** `event_sourcing/event_store.py`
**Description:** Append-only log of events with durable persistence

**Key Responsibilities:**
- Append events to log (immutable, ordered)
- Read events by offset, timestamp, or event type
- Partition events by aggregate ID or key
- Persist events to disk (SQLite, file-based log)
- Support compaction and archival

**Key Methods:**
```python
def append_event(event: Event, partition_key: str) -> EventOffset
def read_events(partition_key: str, from_offset: int, limit: int) -> List[Event]
def read_all_events(from_offset: int = 0) -> Iterator[Event]
def get_partition_offsets() -> Dict[str, int]  # Latest offset per partition
def compact_partition(partition_key: str, snapshot_offset: int) -> None
```

**Schema (msgspec.Struct):**
```python
class Event(msgspec.Struct, kw_only=True, frozen=True):
    event_id: str  # UUID
    event_type: str  # "OrderCreated", "ItemAdded", etc.
    aggregate_id: str  # Order ID, User ID, etc.
    partition_key: str  # For ordering guarantees
    timestamp: float  # Unix timestamp
    payload: Dict[str, Any]  # Event data
    metadata: Dict[str, str]  # Causality tracking, correlation IDs
    version: int  # Schema version

class EventOffset(msgspec.Struct, kw_only=True, frozen=True):
    partition_key: str
    offset: int  # Sequential number within partition
    global_offset: int  # Global sequence number
```

#### Component 2: Event Publisher
**Type:** Write path
**File Path:** `event_sourcing/publisher.py`
**Description:** Publishes events to the event store with validation

**Key Responsibilities:**
- Validate event schemas
- Assign partition keys (hash-based or explicit)
- Assign sequential offsets
- Ensure durability (fsync before acknowledging)
- Emit metrics (events published, latency)

**Key Methods:**
```python
def publish(event: Event) -> EventOffset
def publish_batch(events: List[Event]) -> List[EventOffset]
def validate_event_schema(event: Event) -> bool
def compute_partition_key(aggregate_id: str, num_partitions: int) -> str
```

**Schema:**
```python
class PublishResult(msgspec.Struct, kw_only=True):
    event_id: str
    offset: EventOffset
    success: bool
    error_message: Optional[str]
    latency_ms: float
```

#### Component 3: Event Consumer
**Type:** Read path
**File Path:** `event_sourcing/consumer.py`
**Description:** Reads events from event store and processes them

**Key Responsibilities:**
- Subscribe to event streams by partition or event type
- Maintain consumer offset (last processed event)
- Commit offsets after successful processing
- Handle processing failures (retry, dead letter queue)
- Support at-least-once delivery semantics

**Key Methods:**
```python
def subscribe(partition_keys: List[str], from_offset: int = 0) -> None
def poll(timeout_ms: int) -> List[Event]
def commit_offset(partition_key: str, offset: int) -> None
def get_current_offset(partition_key: str) -> int
def handle_failure(event: Event, error: Exception) -> None
```

**Schema:**
```python
class ConsumerConfig(msgspec.Struct, kw_only=True):
    consumer_group: str  # Logical consumer group name
    partition_assignment: List[str]  # Assigned partitions
    auto_commit: bool  # Auto-commit offsets after processing
    max_poll_records: int  # Max events per poll
    retry_policy: RetryPolicy

class RetryPolicy(msgspec.Struct, kw_only=True):
    max_retries: int
    backoff_ms: int
    dead_letter_queue_enabled: bool
```

#### Component 4: Snapshot Manager
**Type:** Optimization layer
**File Path:** `event_sourcing/snapshot.py`
**Description:** Creates and restores snapshots of aggregate state

**Key Responsibilities:**
- Serialize aggregate state at specific offsets
- Store snapshots with metadata (offset, timestamp)
- Restore state from snapshot + replay delta events
- Implement snapshot compaction (remove old snapshots)
- Support multiple snapshot strategies (time-based, event-count-based)

**Key Methods:**
```python
def create_snapshot(aggregate_id: str, state: Any, offset: int) -> SnapshotId
def load_snapshot(aggregate_id: str) -> Optional[Snapshot]
def restore_state(aggregate_id: str, target_offset: int) -> Any
def list_snapshots(aggregate_id: str) -> List[SnapshotMetadata]
def prune_old_snapshots(aggregate_id: str, keep_count: int) -> None
```

**Schema:**
```python
class Snapshot(msgspec.Struct, kw_only=True):
    snapshot_id: str
    aggregate_id: str
    offset: int  # Event offset when snapshot was taken
    timestamp: float
    state_data: bytes  # Serialized state (msgpack)
    schema_version: int

class SnapshotMetadata(msgspec.Struct, kw_only=True):
    snapshot_id: str
    aggregate_id: str
    offset: int
    timestamp: float
    size_bytes: int
```

#### Component 5: Consumer Group Coordinator
**Type:** Distributed coordination
**File Path:** `event_sourcing/consumer_group.py`
**Description:** Manages consumer group membership and partition assignment

**Key Responsibilities:**
- Track active consumers in a group
- Assign partitions to consumers (load balancing)
- Rebalance partitions when consumers join/leave
- Maintain consumer heartbeats
- Handle consumer failures (reassign partitions)

**Key Methods:**
```python
def register_consumer(consumer_id: str, group_id: str) -> PartitionAssignment
def heartbeat(consumer_id: str) -> bool
def unregister_consumer(consumer_id: str) -> None
def get_partition_assignment(consumer_id: str) -> List[str]
def trigger_rebalance(group_id: str) -> Dict[str, List[str]]
```

**Schema:**
```python
class ConsumerRegistration(msgspec.Struct, kw_only=True):
    consumer_id: str
    group_id: str
    registered_at: float
    last_heartbeat: float
    assigned_partitions: List[str]

class PartitionAssignment(msgspec.Struct, kw_only=True):
    consumer_id: str
    partitions: List[str]
    generation: int  # Rebalance generation number
```

#### Component 6: Dead Letter Queue
**Type:** Error handling
**File Path:** `event_sourcing/dead_letter_queue.py`
**Description:** Stores events that failed processing after max retries

**Key Responsibilities:**
- Store failed events with error context
- Track retry attempts and error messages
- Support manual reprocessing of dead letter events
- Provide visibility into failure patterns

**Key Methods:**
```python
def add_failed_event(event: Event, error: Exception, retry_count: int) -> None
def list_failed_events(limit: int, offset: int) -> List[FailedEvent]
def retry_event(failed_event_id: str) -> PublishResult
def delete_failed_event(failed_event_id: str) -> None
def get_failure_stats() -> FailureStats
```

**Schema:**
```python
class FailedEvent(msgspec.Struct, kw_only=True):
    failed_event_id: str
    original_event: Event
    error_message: str
    error_type: str
    retry_count: int
    first_failed_at: float
    last_failed_at: float
    consumer_id: str

class FailureStats(msgspec.Struct, kw_only=True):
    total_failures: int
    failures_by_type: Dict[str, int]
    failures_by_consumer: Dict[str, int]
```

### 2.2 Component Dependency Graph

```
Event (schema)
    ↓
EventStore ← Publisher
    ↓         ↓
SnapshotManager  Consumer ← ConsumerGroupCoordinator
    ↓                ↓
    └────────→ DeadLetterQueue
```

**Wave 0 (No Dependencies):**
- Event schema
- EventOffset schema

**Wave 1 (Depends on Wave 0):**
- EventStore
- Publisher

**Wave 2 (Depends on Wave 1):**
- Consumer
- SnapshotManager
- DeadLetterQueue

**Wave 3 (Depends on Wave 2):**
- ConsumerGroupCoordinator

---

## 3. EVENT SCHEMA AND VERSIONING

### 3.1 Event Schema Evolution

**Challenge:** Events are immutable, but application schemas evolve

**Strategy:** Schema versioning with backward compatibility

**Example:**
```python
# Version 1
class OrderCreatedV1(msgspec.Struct):
    order_id: str
    customer_id: str
    total: float

# Version 2 (add shipping address)
class OrderCreatedV2(msgspec.Struct):
    order_id: str
    customer_id: str
    total: float
    shipping_address: str  # NEW FIELD

# Upcaster: V1 -> V2
def upcast_order_created_v1_to_v2(event_v1: OrderCreatedV1) -> OrderCreatedV2:
    return OrderCreatedV2(
        order_id=event_v1.order_id,
        customer_id=event_v1.customer_id,
        total=event_v1.total,
        shipping_address="UNKNOWN"  # Default for old events
    )
```

**Implementation:**
- Store `schema_version` in Event metadata
- Maintain upcaster registry: `{(event_type, from_version, to_version): upcaster_fn}`
- Apply upcasters during event deserialization

### 3.2 Event Type Hierarchy

**Domain Events (Business Logic):**
- `OrderCreated`, `OrderSubmitted`, `OrderShipped`
- `PaymentProcessed`, `PaymentFailed`
- `InventoryReserved`, `InventoryReleased`

**System Events (Infrastructure):**
- `SnapshotCreated`
- `ConsumerJoined`, `ConsumerLeft`
- `PartitionRebalanced`

**Failure Events:**
- `EventProcessingFailed`
- `DeadLetterQueued`

---

## 4. MESSAGE ORDERING GUARANTEES

### 4.1 Ordering Semantics

**Per-Partition Ordering (STRICT):**
- All events with same `partition_key` are totally ordered
- Events arrive in sequence: E1 → E2 → E3
- Consumer sees same order: E1 → E2 → E3

**Cross-Partition Ordering (NONE):**
- Events in different partitions have no ordering guarantee
- Partition A: [E1, E2], Partition B: [E3, E4]
- Consumer may see: E1, E3, E2, E4 (any interleaving)

**Global Ordering (WEAK):**
- `global_offset` provides weak global order
- Only useful for debugging, not application logic

### 4.2 Partition Key Selection

**Strategies:**
1. **Aggregate ID:** All events for same order go to same partition
   ```python
   partition_key = f"order:{order_id}"
   ```

2. **Hash-Based:** Distribute load across N partitions
   ```python
   partition_key = f"partition_{hash(order_id) % num_partitions}"
   ```

3. **Custom:** Business logic determines partitioning
   ```python
   partition_key = f"region:{order.region}"  # Regional ordering
   ```

### 4.3 Ordering Implementation

**EventStore Guarantees:**
```python
def append_event(event: Event, partition_key: str) -> EventOffset:
    with self.partition_locks[partition_key]:  # Mutex per partition
        current_offset = self.get_partition_offset(partition_key)
        next_offset = current_offset + 1

        event_record = EventRecord(
            event=event,
            partition_key=partition_key,
            offset=next_offset,
            timestamp=time.time()
        )

        self.persist_event(event_record)  # Atomic write
        self.update_partition_offset(partition_key, next_offset)

        return EventOffset(
            partition_key=partition_key,
            offset=next_offset,
            global_offset=self.global_offset_counter.increment()
        )
```

**Key Properties:**
- Per-partition mutex prevents race conditions
- Offsets are sequential and gap-free
- Persist before updating offset (durability)

---

## 5. EVENT REPLAY AND STATE RECONSTRUCTION

### 5.1 Replay Algorithm

**Goal:** Reconstruct aggregate state by replaying all events

**Pseudocode:**
```python
def rebuild_state(aggregate_id: str) -> OrderState:
    events = event_store.read_events(
        partition_key=f"order:{aggregate_id}",
        from_offset=0
    )

    state = OrderState()  # Initial empty state

    for event in events:
        state = apply_event(state, event)  # Pure function

    return state
```

**Event Handlers:**
```python
def apply_event(state: OrderState, event: Event) -> OrderState:
    match event.event_type:
        case "OrderCreated":
            return OrderState(
                order_id=event.payload["order_id"],
                status="CREATED",
                items=[],
                total=0.0
            )
        case "ItemAdded":
            return state.with_item(event.payload["item"])
        case "OrderSubmitted":
            return state.with_status("SUBMITTED")
        # ... more handlers
```

**Properties:**
- Event handlers are **pure functions** (no side effects)
- Same events always produce same state (deterministic)
- Can replay from any offset (idempotent)

### 5.2 Snapshot-Based Replay (Optimization)

**Goal:** Reduce replay time by starting from snapshot

**Algorithm:**
```python
def rebuild_state_optimized(aggregate_id: str) -> OrderState:
    snapshot = snapshot_manager.load_snapshot(aggregate_id)

    if snapshot:
        state = deserialize_state(snapshot.state_data)
        from_offset = snapshot.offset + 1
    else:
        state = OrderState()  # Initial state
        from_offset = 0

    events = event_store.read_events(
        partition_key=f"order:{aggregate_id}",
        from_offset=from_offset
    )

    for event in events:
        state = apply_event(state, event)

    return state
```

**Performance:**
- Full replay: 1M events × 0.1ms = 100 seconds
- Snapshot replay: Latest snapshot + 1K events = 0.1 seconds
- **1000x speedup!**

### 5.3 Snapshot Creation Strategy

**Time-Based:**
```python
# Create snapshot every hour
if time.time() - last_snapshot_time > 3600:
    snapshot_manager.create_snapshot(aggregate_id, state, current_offset)
```

**Event-Count-Based:**
```python
# Create snapshot every 10,000 events
if events_since_last_snapshot >= 10_000:
    snapshot_manager.create_snapshot(aggregate_id, state, current_offset)
```

**Hybrid:**
```python
# Whichever comes first
if (events_since_last_snapshot >= 10_000 or
    time.time() - last_snapshot_time > 3600):
    snapshot_manager.create_snapshot(aggregate_id, state, current_offset)
```

---

## 6. CONSUMER GROUP MANAGEMENT

### 6.1 Partition Assignment Algorithm

**Goal:** Distribute partitions evenly across consumers

**Strategy:** Range Assignment (simple, deterministic)

```python
def assign_partitions(consumers: List[str], partitions: List[str]) -> Dict[str, List[str]]:
    """Assign partitions to consumers using range-based strategy"""
    num_consumers = len(consumers)
    num_partitions = len(partitions)

    assignments = {}
    partitions_per_consumer = num_partitions // num_consumers
    remainder = num_partitions % num_consumers

    partition_idx = 0
    for i, consumer in enumerate(consumers):
        count = partitions_per_consumer + (1 if i < remainder else 0)
        assignments[consumer] = partitions[partition_idx:partition_idx + count]
        partition_idx += count

    return assignments
```

**Example:**
- 10 partitions: [P0, P1, ..., P9]
- 3 consumers: [C0, C1, C2]
- Assignment:
  - C0: [P0, P1, P2, P3] (4 partitions)
  - C1: [P4, P5, P6] (3 partitions)
  - C2: [P7, P8, P9] (3 partitions)

### 6.2 Rebalancing Protocol

**Trigger:** Consumer joins or leaves group

**Steps:**
1. Coordinator detects consumer change (heartbeat timeout or new registration)
2. Increment rebalance generation number
3. Compute new partition assignment
4. Send assignment to all active consumers
5. Consumers commit current offsets and start consuming new partitions

**Challenge:** Avoiding duplicate processing during rebalance

**Solution:** Fencing with generation numbers
```python
def commit_offset(consumer_id: str, partition_key: str, offset: int, generation: int):
    current_gen = self.get_consumer_generation(consumer_id)

    if generation < current_gen:
        raise FencedConsumerError("Consumer generation is stale")

    # Commit offset only if generation matches
    self.offsets[partition_key] = offset
```

### 6.3 Heartbeat Mechanism

**Protocol:**
- Consumer sends heartbeat every N seconds (e.g., 5s)
- Coordinator tracks `last_heartbeat` timestamp
- If no heartbeat for M seconds (e.g., 15s), mark consumer as dead
- Trigger rebalance to reassign partitions

**Implementation:**
```python
async def heartbeat_loop(consumer: Consumer):
    while consumer.is_active:
        try:
            coordinator.heartbeat(consumer.id)
            await asyncio.sleep(5)  # Heartbeat interval
        except Exception as e:
            logger.error(f"Heartbeat failed: {e}")
            consumer.shutdown()
```

---

## 7. DEAD LETTER QUEUE HANDLING

### 7.1 Failure Detection

**Retry Policy:**
```python
class RetryPolicy(msgspec.Struct, kw_only=True):
    max_retries: int = 3
    backoff_ms: int = 1000  # Exponential backoff base
    dead_letter_queue_enabled: bool = True
```

**Retry Logic:**
```python
async def process_event_with_retry(event: Event, handler: Callable, retry_policy: RetryPolicy):
    attempt = 0

    while attempt <= retry_policy.max_retries:
        try:
            await handler(event)
            return  # Success
        except Exception as e:
            attempt += 1
            if attempt > retry_policy.max_retries:
                if retry_policy.dead_letter_queue_enabled:
                    dlq.add_failed_event(event, e, attempt)
                raise
            else:
                backoff = retry_policy.backoff_ms * (2 ** (attempt - 1))
                await asyncio.sleep(backoff / 1000)
```

### 7.2 Dead Letter Queue Operations

**Add Failed Event:**
```python
def add_failed_event(event: Event, error: Exception, retry_count: int):
    failed_event = FailedEvent(
        failed_event_id=str(uuid.uuid4()),
        original_event=event,
        error_message=str(error),
        error_type=type(error).__name__,
        retry_count=retry_count,
        first_failed_at=time.time(),
        last_failed_at=time.time(),
        consumer_id=consumer.id
    )
    self.failed_events.append(failed_event)
```

**Retry from DLQ:**
```python
def retry_failed_event(failed_event_id: str):
    failed_event = self.get_failed_event(failed_event_id)

    # Republish to event stream
    publisher.publish(failed_event.original_event)

    # Remove from DLQ
    self.delete_failed_event(failed_event_id)
```

### 7.3 Failure Pattern Analysis

**Example Queries:**
```python
# Most common failure types
failure_stats.failures_by_type
# {"ValueError": 42, "TimeoutError": 15, "DatabaseError": 8}

# Consumers with highest failure rates
failure_stats.failures_by_consumer
# {"consumer-1": 30, "consumer-2": 20, "consumer-3": 15}
```

**Actionable Insights:**
- High `ValueError` count → Schema validation issue
- High `TimeoutError` count → Downstream service slow
- High failures on specific consumer → Bug in that instance

---

## 8. TEST SCENARIOS

### 8.1 Basic Event Flow Scenarios

#### Scenario 1: Publish and Consume Single Event
**Setup:**
- Empty event store
- Single consumer subscribed to partition "order:123"

**Steps:**
1. Publish `OrderCreated` event with `partition_key="order:123"`
2. Consumer polls for events
3. Consumer processes event
4. Consumer commits offset

**Expected Behavior:**
```python
event = Event(
    event_id="evt-001",
    event_type="OrderCreated",
    aggregate_id="order-123",
    partition_key="order:123",
    payload={"order_id": "order-123", "customer_id": "cust-456"}
)

offset = publisher.publish(event)
assert offset.offset == 1
assert offset.partition_key == "order:123"

events = consumer.poll(timeout_ms=1000)
assert len(events) == 1
assert events[0].event_id == "evt-001"

consumer.commit_offset("order:123", 1)
assert consumer.get_current_offset("order:123") == 1
```

#### Scenario 2: Event Ordering Within Partition
**Setup:**
- Publish 3 events to same partition: E1, E2, E3

**Expected Behavior:**
```python
events = [
    Event(event_id="e1", partition_key="p1", ...),
    Event(event_id="e2", partition_key="p1", ...),
    Event(event_id="e3", partition_key="p1", ...),
]

for event in events:
    publisher.publish(event)

consumed = consumer.poll(timeout_ms=1000)
assert [e.event_id for e in consumed] == ["e1", "e2", "e3"]
```

#### Scenario 3: Cross-Partition Independence
**Setup:**
- Publish events to 2 partitions: P1 and P2

**Expected Behavior:**
```python
# Events can be consumed in any interleaving across partitions
publisher.publish(Event(event_id="p1-e1", partition_key="p1"))
publisher.publish(Event(event_id="p2-e1", partition_key="p2"))
publisher.publish(Event(event_id="p1-e2", partition_key="p1"))

consumed = consumer.poll(timeout_ms=1000)
# No ordering guarantee between p1 and p2 events
# But p1-e1 MUST come before p1-e2
```

### 8.2 Snapshot and Replay Scenarios

#### Scenario 4: Full Replay from Beginning
**Setup:**
- 1000 events in event store
- No snapshots

**Steps:**
1. Rebuild state from offset 0
2. Apply all 1000 events

**Expected Behavior:**
```python
state = rebuild_state("order-123")

# Verify final state matches expected
assert state.status == "SHIPPED"
assert len(state.items) == 5
assert state.total == 299.99
```

**Assertions:**
```python
assert replay_time < 1.0  # 1 second for 1000 events
```

#### Scenario 5: Snapshot-Based Replay
**Setup:**
- 10,000 events in event store
- Snapshot at offset 9000

**Steps:**
1. Load snapshot (offset 9000)
2. Replay events 9001-10000 (1000 events)

**Expected Behavior:**
```python
state = rebuild_state_optimized("order-123")

# Same final state as full replay
assert state == full_replay_state
```

**Assertions:**
```python
assert snapshot_replay_time < full_replay_time / 5
```

#### Scenario 6: Snapshot Creation
**Setup:**
- Consumer processed 10,000 events
- Snapshot policy: every 10,000 events

**Expected Behavior:**
```python
snapshot = snapshot_manager.create_snapshot(
    aggregate_id="order-123",
    state=current_state,
    offset=10_000
)

assert snapshot.offset == 10_000
assert snapshot_manager.load_snapshot("order-123") == snapshot
```

### 8.3 Consumer Group Scenarios

#### Scenario 7: Single Consumer in Group
**Setup:**
- 4 partitions: P0, P1, P2, P3
- 1 consumer in group "billing"

**Expected Behavior:**
```python
coordinator.register_consumer("consumer-1", "billing")
assignment = coordinator.get_partition_assignment("consumer-1")

assert set(assignment) == {"P0", "P1", "P2", "P3"}  # All partitions
```

#### Scenario 8: Multiple Consumers in Group
**Setup:**
- 4 partitions: P0, P1, P2, P3
- 2 consumers in group "billing"

**Expected Behavior:**
```python
coordinator.register_consumer("consumer-1", "billing")
coordinator.register_consumer("consumer-2", "billing")

assignment_1 = coordinator.get_partition_assignment("consumer-1")
assignment_2 = coordinator.get_partition_assignment("consumer-2")

# Partitions split evenly
assert len(assignment_1) == 2
assert len(assignment_2) == 2
assert set(assignment_1).isdisjoint(set(assignment_2))
assert set(assignment_1) | set(assignment_2) == {"P0", "P1", "P2", "P3"}
```

#### Scenario 9: Consumer Failure and Rebalance
**Setup:**
- 3 consumers in group
- Consumer-2 crashes (heartbeat stops)

**Steps:**
1. Coordinator detects missed heartbeat after 15s
2. Mark consumer-2 as dead
3. Trigger rebalance
4. Reassign consumer-2's partitions to consumer-1 and consumer-3

**Expected Behavior:**
```python
# Before failure
assignment_2 = coordinator.get_partition_assignment("consumer-2")
assert len(assignment_2) > 0

# After failure (15s timeout)
await asyncio.sleep(15)
coordinator.heartbeat_check()

# Consumer-2 partitions redistributed
assignment_1_new = coordinator.get_partition_assignment("consumer-1")
assignment_3_new = coordinator.get_partition_assignment("consumer-3")

assert len(assignment_1_new) > len(assignment_1_old)
assert "consumer-2" not in coordinator.active_consumers
```

### 8.4 Dead Letter Queue Scenarios

#### Scenario 10: Event Processing Failure
**Setup:**
- Event handler throws exception
- Retry policy: max 3 retries

**Steps:**
1. Consumer polls event
2. Handler fails 3 times
3. Event moved to DLQ

**Expected Behavior:**
```python
@retry_on_failure(max_retries=3)
async def buggy_handler(event: Event):
    raise ValueError("Processing failed")

# Process event
await process_event_with_retry(event, buggy_handler, retry_policy)

# Event in DLQ after max retries
failed_events = dlq.list_failed_events(limit=10)
assert len(failed_events) == 1
assert failed_events[0].original_event.event_id == event.event_id
assert failed_events[0].retry_count == 3
```

#### Scenario 11: DLQ Retry
**Setup:**
- Failed event in DLQ
- Bug fixed in handler

**Steps:**
1. Retry failed event from DLQ
2. Event reprocessed successfully

**Expected Behavior:**
```python
failed_event_id = failed_events[0].failed_event_id

# Retry from DLQ
dlq.retry_event(failed_event_id)

# Event removed from DLQ
remaining_failed = dlq.list_failed_events(limit=10)
assert len(remaining_failed) == 0
```

### 8.5 Eventual Consistency Scenarios

#### Scenario 12: Order Processing Across Services
**Setup:**
- 3 consumers in different services: inventory, billing, shipping
- All consume same `OrderSubmitted` event

**Expected Behavior:**
```python
# Publish event
event = Event(event_type="OrderSubmitted", aggregate_id="order-123")
publisher.publish(event)

# All consumers eventually process event
await asyncio.sleep(1)  # Allow propagation

assert inventory_service.has_processed("order-123")
assert billing_service.has_processed("order-123")
assert shipping_service.has_processed("order-123")
```

**Timing:**
- Events may be processed in different orders across services
- But all services eventually reach consistent state

#### Scenario 13: Idempotent Event Processing
**Setup:**
- Same event delivered twice (at-least-once semantics)

**Expected Behavior:**
```python
# Process event twice
handler(event)  # First time: create record
handler(event)  # Second time: idempotent (no duplicate)

# State unchanged
assert database.count_orders() == 1  # Not 2!
```

**Implementation Strategy:**
- Check event ID before processing
- Skip if already processed (idempotency key)

---

## 9. TESTING STRATEGY

### 9.1 Unit Tests

**EventStore (12 tests):**
- `test_append_event_creates_record`
- `test_append_event_assigns_sequential_offset`
- `test_read_events_by_partition`
- `test_read_events_from_offset`
- `test_read_all_events_iterator`
- `test_get_partition_offsets_accurate`
- `test_compact_partition_removes_old_events`
- `test_concurrent_append_preserves_order`
- `test_persist_survives_process_restart`
- `test_partition_isolation`
- `test_event_immutability`
- `test_large_event_payload`

**Publisher (8 tests):**
- `test_publish_single_event`
- `test_publish_batch_events`
- `test_validate_event_schema_success`
- `test_validate_event_schema_failure`
- `test_compute_partition_key_hash_based`
- `test_compute_partition_key_explicit`
- `test_publish_latency_tracking`
- `test_publish_durability_fsync`

**Consumer (10 tests):**
- `test_subscribe_to_partition`
- `test_poll_returns_events`
- `test_poll_timeout_returns_empty`
- `test_commit_offset_updates_position`
- `test_auto_commit_after_poll`
- `test_handle_failure_retry_logic`
- `test_handle_failure_dlq_after_max_retries`
- `test_consume_from_specific_offset`
- `test_consumer_offset_persistence`
- `test_multiple_consumers_independent_offsets`

**SnapshotManager (10 tests):**
- `test_create_snapshot_stores_state`
- `test_load_snapshot_returns_latest`
- `test_restore_state_from_snapshot`
- `test_restore_state_with_delta_events`
- `test_list_snapshots_for_aggregate`
- `test_prune_old_snapshots`
- `test_snapshot_serialization_msgpack`
- `test_snapshot_schema_versioning`
- `test_snapshot_creation_timing_policy`
- `test_snapshot_creation_event_count_policy`

**ConsumerGroupCoordinator (10 tests):**
- `test_register_consumer_assigns_partitions`
- `test_heartbeat_updates_timestamp`
- `test_unregister_consumer_triggers_rebalance`
- `test_partition_assignment_even_distribution`
- `test_rebalance_on_consumer_join`
- `test_rebalance_on_consumer_leave`
- `test_heartbeat_timeout_detection`
- `test_generation_number_increments`
- `test_fencing_stale_consumers`
- `test_multiple_consumer_groups_isolated`

**DeadLetterQueue (8 tests):**
- `test_add_failed_event`
- `test_list_failed_events_pagination`
- `test_retry_event_republishes`
- `test_delete_failed_event`
- `test_get_failure_stats_by_type`
- `test_get_failure_stats_by_consumer`
- `test_failed_event_metadata_complete`
- `test_dlq_persistence`

### 9.2 Integration Tests

**Event Sourcing Integration (10 tests):**
- `test_publish_consume_end_to_end`
- `test_event_ordering_within_partition`
- `test_snapshot_based_replay_correctness`
- `test_consumer_group_load_balancing`
- `test_consumer_failure_rebalance`
- `test_dead_letter_queue_retry_flow`
- `test_schema_evolution_upcasting`
- `test_concurrent_publishers_no_conflict`
- `test_durability_after_crash_recovery`
- `test_idempotent_event_processing`

### 9.3 Property-Based Tests (Hypothesis)

**Event Ordering Property:**
```python
@given(st.lists(st.text(), min_size=10, max_size=100))
def test_event_ordering_preserved(event_ids):
    """Events in same partition maintain order"""
    partition_key = "test-partition"

    # Publish events in order
    for event_id in event_ids:
        publisher.publish(Event(event_id=event_id, partition_key=partition_key))

    # Consume events
    consumed = consumer.poll_all()
    consumed_ids = [e.event_id for e in consumed]

    # Property: Order preserved
    assert consumed_ids == event_ids
```

**Snapshot Replay Equivalence:**
```python
@given(st.lists(st.dictionaries(st.text(), st.integers()), min_size=50, max_size=200))
def test_snapshot_replay_equivalence(events):
    """Snapshot + delta = full replay"""
    aggregate_id = "test-aggregate"

    # Full replay
    full_state = rebuild_state(aggregate_id)

    # Snapshot + delta replay
    snapshot = snapshot_manager.load_snapshot(aggregate_id)
    snapshot_state = rebuild_state_optimized(aggregate_id)

    # Property: States must be identical
    assert snapshot_state == full_state
```

---

## 10. PARAGON-SPECIFIC INTEGRATION

### 10.1 Graph Database Integration

**Map Event Sourcing to ParagonDB:**
- Events → `NodeType.EVENT`
- Snapshots → `NodeType.SNAPSHOT`
- Consumers → `NodeType.AGENT`
- Causal relationships → `EdgeType.CAUSES` (event E1 caused E2)
- Snapshot references → `EdgeType.CAPTURES` (snapshot captures state at offset N)

**Benefits:**
- Teleology: trace event lineage (which events caused this state?)
- Merkle hashing: detect event stream tampering
- Graph queries: find all events of type X in time range Y

### 10.2 Orchestrator Integration

**TDD Cycle Mapping:**
1. **DIALECTIC:** Clarify consistency model (eventual vs strong)
2. **RESEARCH:** Study event sourcing patterns (Kafka, EventStore)
3. **PLAN:** Architect creates component breakdown
4. **BUILD:** Builder implements EventStore, Publisher, Consumer
5. **TEST:** Tester verifies ordering, durability, replay correctness

**Checkpoint Integration:**
- Store event offsets in SqliteSaver
- Resume consumer from last committed offset after crash

### 10.3 Rerun Visualization

**Visualizations:**
- Event stream timeline (events as points on timeline)
- Partition waterfall (show offsets per partition over time)
- Consumer lag chart (how far behind is each consumer?)
- Snapshot creation events (vertical lines on timeline)
- Dead letter queue heatmap (failure density over time)

---

## 11. EXTENSION POINTS

### 11.1 Advanced Features (Out of Scope for MVP)

**Distributed Event Store:**
- Replicate events across multiple nodes
- Raft consensus for partition leadership
- Quorum writes for durability

**Event Compaction:**
- Retain only latest state per aggregate
- Remove intermediate events after snapshot
- Save storage space for high-volume streams

**Query Projections:**
- Materialized views from event streams
- Support complex queries (find all orders by customer)
- Incremental updates on new events

### 11.2 Integration with External Systems

**Kafka Integration:**
- Publish events to Kafka topics
- Consume from Kafka with consumer groups
- Use Kafka's offset management

**Change Data Capture (CDC):**
- Stream database changes as events
- Debezium connector for PostgreSQL/MySQL
- Bi-directional sync with event store

---

## 12. EVALUATION CRITERIA

### 12.1 Orchestrator Evaluation

**How well does the orchestrator handle this problem?**
- Does it recognize the append-only, immutable nature?
- Does it identify ordering requirements early?
- Does it design for failure recovery (DLQ, retries)?
- Does it implement snapshot optimization?
- Does it handle schema evolution?

### 12.2 Code Quality Metrics

**Generated Code Quality:**
- Test coverage ≥ 90%
- Correct use of async/await for I/O
- Proper resource cleanup (file handles, locks)
- Schema validation with msgspec
- No data loss scenarios

### 12.3 Performance Validation

**Benchmark Targets:**
```python
# Protocol Alpha extension
def test_event_sourcing_performance():
    """Test event sourcing system performance"""
    # Ingestion throughput
    assert events_per_second > 10_000

    # Consumer latency
    assert p99_latency_ms < 1.0

    # Snapshot replay speedup
    assert snapshot_replay_time < full_replay_time / 10
```

---

## 13. IMPLEMENTATION CHECKLIST

### Phase 1: Foundation (Day 1-2)
- [ ] Define all msgspec schemas (Event, EventOffset, Snapshot, etc.)
- [ ] Implement EventStore with append-only log
- [ ] Implement Publisher with partition assignment
- [ ] Unit tests for EventStore and Publisher (20 tests)

### Phase 2: Consumer (Day 2-3)
- [ ] Implement Consumer with offset tracking
- [ ] Implement RetryPolicy and failure handling
- [ ] Implement DeadLetterQueue
- [ ] Unit tests (18 tests)
- [ ] Integration test: publish → consume flow

### Phase 3: Snapshots (Day 3-4)
- [ ] Implement SnapshotManager with serialization
- [ ] Implement snapshot creation policies
- [ ] Implement optimized replay algorithm
- [ ] Unit tests (10 tests)
- [ ] Integration test: snapshot-based replay

### Phase 4: Consumer Groups (Day 4-5)
- [ ] Implement ConsumerGroupCoordinator
- [ ] Implement partition assignment algorithm
- [ ] Implement heartbeat and rebalancing
- [ ] Unit tests (10 tests)
- [ ] Integration test: multi-consumer scenario

### Phase 5: Integration & Testing (Day 5-6)
- [ ] Integrate with ParagonDB
- [ ] Add Rerun visualization hooks
- [ ] Property-based tests (Hypothesis)
- [ ] Performance benchmarks
- [ ] End-to-end test: full TDD cycle
- [ ] Documentation and examples

---

## 14. EXPECTED LEARNING OUTCOMES

### For the Orchestrator:
1. **Immutability:** Understanding append-only data structures
2. **Ordering:** Reasoning about partial vs total order
3. **Failure handling:** Designing for retries and recovery
4. **Performance optimization:** Snapshot vs full replay tradeoffs

### For the Paragon System:
1. **Event-driven architecture:** Modeling events in graph
2. **Async patterns:** Efficient I/O-bound task scheduling
3. **Durability:** Testing crash recovery scenarios
4. **Schema evolution:** Handling versioning over time

---

## 15. DELIVERABLES

### Code Artifacts:
- `event_sourcing/` module with 6 Python files
- `tests/event_sourcing/` with 70+ unit tests
- `tests/integration/test_event_sourcing.py` with 10 integration tests
- Example order processing system in `examples/order_management/`

### Documentation:
- API documentation (docstrings)
- User guide (how to use event sourcing)
- Architecture diagram (component interactions)
- Event schema examples

### Metrics:
- Test coverage report (≥90% target)
- Performance benchmark results
- Failure recovery time measurements

---

## APPENDIX A: Reference Implementations

### Real-World Event Sourcing Systems:
1. **Kafka:** Distributed event streaming platform
2. **EventStoreDB:** Purpose-built event sourcing database
3. **AWS EventBridge:** Serverless event bus
4. **Apache Pulsar:** Multi-tenant event streaming
5. **NATS JetStream:** Cloud-native messaging with persistence

### Key Learnings:
- Kafka's offset model maps directly to EventStore design
- EventStoreDB's projection system is similar to SnapshotManager
- Pulsar's tiered storage shows compaction strategies

---

## APPENDIX B: Prompt Template for Orchestrator

**When submitting this problem to Paragon:**

```markdown
I need an event sourcing system for my distributed order management application.

REQUIREMENTS:
1. Store all state changes as immutable events in append-only log
2. Support multiple consumers reading events (inventory, billing, shipping)
3. Replay events to rebuild state after crashes
4. Create snapshots for fast recovery (don't replay 1M events every time)
5. Handle consumer failures gracefully (retry + dead letter queue)
6. Guarantee event ordering within same order ID
7. Support consumer groups for load balancing

EXAMPLE WORKFLOW:
- Order created → OrderCreated event
- Item added → ItemAdded event
- Payment processed → PaymentProcessed event
- Order shipped → OrderShipped event
- If billing service crashes, replay events from last checkpoint
- If event processing fails 3 times, send to dead letter queue

CONSTRAINTS:
- Events are immutable (no updates or deletes)
- Must handle 10,000 events/sec
- Consumer lag must be < 1 second p99
- Snapshot replay must be 10x faster than full replay
- No data loss (durability guarantee)

Please implement this system using the Paragon TDD workflow.
```

**Expected Orchestrator Behavior:**
1. DIALECTIC identifies ordering and consistency questions
2. RESEARCH investigates event sourcing patterns
3. ARCHITECT creates component breakdown (matches Section 2.1)
4. BUILDER generates code with msgspec schemas
5. TESTER verifies all scenarios from Section 8

---

**END OF RESEARCH DOCUMENT**
