# GOLDEN SET PROBLEM 5: Schema Migration Engine

**Problem ID:** `GOLDEN-005`
**Category:** Complex Orchestration - Database Evolution
**Difficulty:** High
**Estimated Implementation Time:** 4-6 days
**Date Created:** 2025-12-07

---

## EXECUTIVE SUMMARY

Build a database schema migration engine that generates migration scripts, validates schema compatibility, and supports rollback operations. This problem tests the orchestrator's ability to:
- Implement graph-based diff algorithms for schema comparison
- Generate executable migration scripts from schema differences
- Detect breaking vs non-breaking changes
- Design safe rollback strategies with data preservation
- Support multiple database backends (PostgreSQL, MySQL, SQLite)
- Handle migration chains with dependency tracking

This is a canonical example of a system requiring careful state transitions, backward compatibility analysis, and data safety guarantees.

---

## 1. PROBLEM STATEMENT

### 1.1 User Requirement (What a User Would Submit)

```
I need a schema migration engine that:
1. Compares two database schemas and generates a diff
2. Creates SQL migration scripts to transform schema A → schema B
3. Detects breaking changes (drop column, change type) vs safe changes (add column)
4. Generates rollback scripts for every migration
5. Executes migrations in correct dependency order (foreign keys, indexes)
6. Validates data compatibility before applying destructive changes
7. Supports multiple databases (PostgreSQL, MySQL, SQLite)
8. Tracks migration history and prevents re-applying migrations

USE CASE:
I have a production database with 100 tables. I want to:
- Add a new column 'email_verified' to 'users' table (safe)
- Change 'price' column from INT to DECIMAL (breaking - needs data migration)
- Drop unused 'legacy_flags' table (destructive)
- Add index on 'orders.created_at' (performance optimization)

The system should:
- Generate migration script with data preservation
- Warn me about breaking changes
- Create rollback script to undo migration
- Execute safely with transaction support
```

### 1.2 Success Criteria

**Functional Requirements:**
- Schema diff detects all differences between schemas
- Migration scripts are syntactically valid SQL
- Breaking change detection is 100% accurate
- Rollback scripts restore exact previous state
- Migration dependencies are correctly ordered
- Supports PostgreSQL, MySQL, SQLite dialects

**Quality Requirements:**
- All graph invariants maintained
- Test coverage ≥ 90%
- Migration correctness: 100% (no data loss)
- Diff accuracy: 100% (no missed changes)
- Performance: Diff 1000-table schema in <1 second
- Safety: All destructive operations require explicit confirmation

---

## 2. CORE COMPONENTS TO IMPLEMENT

### 2.1 Component Breakdown

#### Component 1: Schema Parser
**Type:** Core data extraction
**File Path:** `schema_migration/schema_parser.py`
**Description:** Parses database schema into structured representation

**Key Responsibilities:**
- Extract schema from live database (INFORMATION_SCHEMA queries)
- Parse schema from SQL DDL files
- Normalize schema representation (database-agnostic)
- Handle vendor-specific syntax (PostgreSQL vs MySQL)

**Key Methods:**
```python
def parse_database(connection: DatabaseConnection) -> Schema
def parse_ddl_file(path: Path) -> Schema
def normalize_schema(raw_schema: Dict) -> Schema
def extract_tables(connection: DatabaseConnection) -> List[Table]
def extract_constraints(connection: DatabaseConnection) -> List[Constraint]
```

**Schema (msgspec.Struct):**
```python
class Column(msgspec.Struct, kw_only=True, frozen=True):
    name: str
    data_type: str  # Normalized: "INTEGER", "VARCHAR", "DECIMAL", etc.
    nullable: bool
    default: Optional[str]
    primary_key: bool
    auto_increment: bool
    max_length: Optional[int]  # For VARCHAR(255)
    precision: Optional[int]  # For DECIMAL(10,2)
    scale: Optional[int]

class Table(msgspec.Struct, kw_only=True, frozen=True):
    name: str
    columns: List[Column]
    primary_key: Optional[List[str]]  # Column names
    indexes: List[Index]
    foreign_keys: List[ForeignKey]
    unique_constraints: List[UniqueConstraint]

class Index(msgspec.Struct, kw_only=True, frozen=True):
    name: str
    table_name: str
    columns: List[str]
    unique: bool
    index_type: str  # "BTREE", "HASH", "GIN", etc.

class ForeignKey(msgspec.Struct, kw_only=True, frozen=True):
    name: str
    table_name: str
    columns: List[str]
    referenced_table: str
    referenced_columns: List[str]
    on_delete: str  # "CASCADE", "SET NULL", "RESTRICT"
    on_update: str

class UniqueConstraint(msgspec.Struct, kw_only=True, frozen=True):
    name: str
    table_name: str
    columns: List[str]

class Schema(msgspec.Struct, kw_only=True):
    database_name: str
    tables: List[Table]
    version: str  # Schema version identifier
```

#### Component 2: Schema Diff Engine
**Type:** Core algorithm
**File Path:** `schema_migration/diff_engine.py`
**Description:** Compares two schemas and generates structured diff

**Key Responsibilities:**
- Identify added, removed, and modified tables
- Identify added, removed, and modified columns
- Detect index and constraint changes
- Compute minimal diff (ignore irrelevant changes)
- Handle name changes (column renames vs drop+add)

**Key Methods:**
```python
def compute_diff(old_schema: Schema, new_schema: Schema) -> SchemaDiff
def diff_tables(old_tables: List[Table], new_tables: List[Table]) -> TableDiff
def diff_columns(old_cols: List[Column], new_cols: List[Column]) -> ColumnDiff
def is_column_rename(old_col: Column, new_col: Column) -> bool
def classify_change_type(change: SchemaChange) -> ChangeType
```

**Schema:**
```python
class SchemaChange(msgspec.Struct, kw_only=True, frozen=True):
    change_type: str  # "ADD_TABLE", "DROP_TABLE", "ADD_COLUMN", "MODIFY_COLUMN", etc.
    change_impact: str  # "SAFE", "BREAKING", "DESTRUCTIVE"
    old_value: Optional[Any]
    new_value: Optional[Any]
    description: str  # Human-readable description

class TableDiff(msgspec.Struct, kw_only=True):
    added_tables: List[Table]
    dropped_tables: List[Table]
    modified_tables: List[str]  # Table names

class ColumnDiff(msgspec.Struct, kw_only=True):
    table_name: str
    added_columns: List[Column]
    dropped_columns: List[Column]
    modified_columns: List[Tuple[Column, Column]]  # (old, new)

class SchemaDiff(msgspec.Struct, kw_only=True):
    old_schema_version: str
    new_schema_version: str
    changes: List[SchemaChange]
    table_diff: TableDiff
    column_diffs: List[ColumnDiff]
    breaking_changes: List[SchemaChange]
    safe_changes: List[SchemaChange]
```

#### Component 3: Migration Script Generator
**Type:** Code generation
**File Path:** `schema_migration/script_generator.py`
**Description:** Generates SQL migration scripts from schema diff

**Key Responsibilities:**
- Convert SchemaChange to SQL statements
- Generate forward migration (old → new)
- Generate rollback migration (new → old)
- Order statements by dependencies (FK constraints after tables)
- Handle database-specific SQL dialects

**Key Methods:**
```python
def generate_migration(diff: SchemaDiff, dialect: str) -> MigrationScript
def generate_rollback(diff: SchemaDiff, dialect: str) -> MigrationScript
def order_statements(statements: List[SQLStatement]) -> List[SQLStatement]
def generate_add_table_sql(table: Table, dialect: str) -> str
def generate_add_column_sql(table: str, column: Column, dialect: str) -> str
def generate_drop_column_sql(table: str, column: str, dialect: str) -> str
```

**Schema:**
```python
class SQLStatement(msgspec.Struct, kw_only=True, frozen=True):
    sql: str
    statement_type: str  # "DDL", "DML", "DCL"
    operation: str  # "CREATE_TABLE", "ALTER_TABLE", "DROP_TABLE", etc.
    dependencies: List[str]  # Table names this statement depends on

class MigrationScript(msgspec.Struct, kw_only=True):
    migration_id: str
    from_version: str
    to_version: str
    statements: List[SQLStatement]
    dialect: str  # "postgresql", "mysql", "sqlite"
    breaking_changes: List[str]  # Warnings
    estimated_duration_seconds: float
```

#### Component 4: Compatibility Checker
**Type:** Validation layer
**File Path:** `schema_migration/compatibility.py`
**Description:** Validates schema changes for breaking changes and data safety

**Key Responsibilities:**
- Classify changes as SAFE, BREAKING, or DESTRUCTIVE
- Detect type compatibility (INT → BIGINT is safe, VARCHAR → INT is breaking)
- Check constraint compatibility (adding NOT NULL is breaking if nulls exist)
- Validate foreign key references
- Generate data migration scripts for type changes

**Key Methods:**
```python
def classify_change(change: SchemaChange) -> ChangeImpact
def is_type_compatible(old_type: str, new_type: str) -> bool
def check_not_null_safety(table: str, column: str, connection: DatabaseConnection) -> bool
def generate_data_migration(change: SchemaChange) -> Optional[SQLStatement]
def validate_foreign_key_references(schema: Schema) -> List[ValidationError]
```

**Schema:**
```python
class ChangeImpact(msgspec.Struct, kw_only=True, frozen=True):
    impact_level: str  # "SAFE", "BREAKING", "DESTRUCTIVE"
    reason: str
    requires_data_migration: bool
    data_loss_risk: bool

class ValidationError(msgspec.Struct, kw_only=True, frozen=True):
    error_type: str  # "ORPHANED_FK", "NULL_VIOLATION", "TYPE_MISMATCH"
    table_name: str
    column_name: Optional[str]
    description: str
```

**Change Classification Rules:**

**SAFE Changes:**
- Add table
- Add nullable column
- Add index
- Widen type (INT → BIGINT, VARCHAR(50) → VARCHAR(100))
- Drop index
- Add foreign key (if data satisfies constraint)

**BREAKING Changes:**
- Add NOT NULL column without default
- Narrow type (BIGINT → INT, VARCHAR(100) → VARCHAR(50))
- Change column type (VARCHAR → INT)
- Add unique constraint (if duplicates exist)
- Modify primary key

**DESTRUCTIVE Changes:**
- Drop table
- Drop column
- Drop foreign key
- Drop unique constraint

#### Component 5: Migration Executor
**Type:** Execution engine
**File Path:** `schema_migration/executor.py`
**Description:** Executes migration scripts against database

**Key Responsibilities:**
- Execute SQL statements in transaction
- Rollback on error
- Validate schema after migration
- Track migration history
- Support dry-run mode (validate without executing)

**Key Methods:**
```python
def execute_migration(script: MigrationScript, connection: DatabaseConnection) -> ExecutionResult
def dry_run_migration(script: MigrationScript, connection: DatabaseConnection) -> ValidationReport
def rollback_migration(script: MigrationScript, connection: DatabaseConnection) -> ExecutionResult
def record_migration_history(migration_id: str, result: ExecutionResult) -> None
def get_migration_history() -> List[MigrationRecord]
```

**Schema:**
```python
class ExecutionResult(msgspec.Struct, kw_only=True):
    success: bool
    executed_statements: List[str]
    failed_statement: Optional[str]
    error_message: Optional[str]
    duration_seconds: float
    rows_affected: int

class MigrationRecord(msgspec.Struct, kw_only=True):
    migration_id: str
    from_version: str
    to_version: str
    applied_at: float  # Timestamp
    success: bool
    duration_seconds: float
    executed_by: str  # User/system identifier
```

#### Component 6: Migration History Tracker
**Type:** Metadata management
**File Path:** `schema_migration/history.py`
**Description:** Tracks applied migrations and prevents re-application

**Key Responsibilities:**
- Store migration history in database table
- Check if migration already applied
- Compute migration path (current → target version)
- Detect migration conflicts (divergent branches)

**Key Methods:**
```python
def initialize_history_table(connection: DatabaseConnection) -> None
def is_migration_applied(migration_id: str) -> bool
def compute_migration_path(current_version: str, target_version: str) -> List[str]
def detect_conflicts(history: List[MigrationRecord]) -> List[ConflictError]
```

**Schema:**
```python
class MigrationPath(msgspec.Struct, kw_only=True):
    start_version: str
    end_version: str
    migration_chain: List[str]  # Ordered migration IDs
    total_steps: int
```

### 2.2 Component Dependency Graph

```
Schema (base schema)
    ↓
SchemaParser → Schema instances
    ↓           ↓
DiffEngine ← Compatibility Checker
    ↓
ScriptGenerator
    ↓
MigrationExecutor → HistoryTracker
```

**Wave 0 (No Dependencies):**
- Schema definitions (Column, Table, Index, etc.)

**Wave 1 (Depends on Wave 0):**
- SchemaParser

**Wave 2 (Depends on Wave 1):**
- DiffEngine
- CompatibilityChecker

**Wave 3 (Depends on Wave 2):**
- ScriptGenerator

**Wave 4 (Depends on Wave 3):**
- MigrationExecutor
- HistoryTracker

---

## 3. SCHEMA DIFF ALGORITHMS

### 3.1 Table-Level Diff

**Algorithm:** Set-based comparison with name matching

```python
def diff_tables(old_schema: Schema, new_schema: Schema) -> TableDiff:
    old_table_names = {t.name for t in old_schema.tables}
    new_table_names = {t.name for t in new_schema.tables}

    added = [t for t in new_schema.tables if t.name not in old_table_names]
    dropped = [t for t in old_schema.tables if t.name not in new_table_names]
    modified = [name for name in old_table_names & new_table_names
                if get_table(old_schema, name) != get_table(new_schema, name)]

    return TableDiff(
        added_tables=added,
        dropped_tables=dropped,
        modified_tables=modified
    )
```

### 3.2 Column-Level Diff

**Algorithm:** Positional and name-based matching

**Challenge:** Distinguish column rename from drop+add

**Heuristic:**
- If column name changed but type/constraints identical → likely rename
- If column name same but type changed → modification
- If column disappears and new column appears → could be rename or separate changes

**Implementation:**
```python
def diff_columns(old_table: Table, new_table: Table) -> ColumnDiff:
    old_cols = {c.name: c for c in old_table.columns}
    new_cols = {c.name: c for c in new_table.columns}

    added = [c for name, c in new_cols.items() if name not in old_cols]
    dropped = [c for name, c in old_cols.items() if name not in new_cols]

    modified = []
    for name in old_cols.keys() & new_cols.keys():
        old_col = old_cols[name]
        new_col = new_cols[name]
        if old_col != new_col:
            modified.append((old_col, new_col))

    # Rename detection heuristic
    for dropped_col in dropped[:]:
        for added_col in added[:]:
            if is_likely_rename(dropped_col, added_col):
                # Treat as rename instead of drop+add
                modified.append((dropped_col, added_col))
                dropped.remove(dropped_col)
                added.remove(added_col)
                break

    return ColumnDiff(
        table_name=new_table.name,
        added_columns=added,
        dropped_columns=dropped,
        modified_columns=modified
    )

def is_likely_rename(old_col: Column, new_col: Column) -> bool:
    """Heuristic: same type + same constraints → likely rename"""
    return (
        old_col.data_type == new_col.data_type and
        old_col.nullable == new_col.nullable and
        old_col.primary_key == new_col.primary_key and
        edit_distance(old_col.name, new_col.name) <= 3  # Similar names
    )
```

### 3.3 Index and Constraint Diff

**Algorithm:** Compare by semantic meaning, not just name

**Example:**
```python
# These are semantically identical despite different names
old_index = Index(name="idx_user_email", columns=["email"], unique=True)
new_index = Index(name="unique_email_idx", columns=["email"], unique=True)

# Diff should recognize this as "no change"
```

**Implementation:**
```python
def diff_indexes(old_indexes: List[Index], new_indexes: List[Index]) -> IndexDiff:
    old_by_signature = {index_signature(idx): idx for idx in old_indexes}
    new_by_signature = {index_signature(idx): idx for idx in new_indexes}

    added = [idx for sig, idx in new_by_signature.items() if sig not in old_by_signature]
    dropped = [idx for sig, idx in old_by_signature.items() if sig not in new_by_signature]

    return IndexDiff(added_indexes=added, dropped_indexes=dropped)

def index_signature(index: Index) -> str:
    """Canonical representation for comparison"""
    return f"{index.table_name}:{','.join(sorted(index.columns))}:{index.unique}"
```

---

## 4. MIGRATION SCRIPT GENERATION

### 4.1 SQL Generation by Change Type

#### Add Table
```python
def generate_add_table_sql(table: Table, dialect: str) -> str:
    columns_sql = []
    for col in table.columns:
        col_def = f"{col.name} {col.data_type}"
        if not col.nullable:
            col_def += " NOT NULL"
        if col.default:
            col_def += f" DEFAULT {col.default}"
        if col.auto_increment and dialect == "mysql":
            col_def += " AUTO_INCREMENT"
        columns_sql.append(col_def)

    if table.primary_key:
        pk_cols = ", ".join(table.primary_key)
        columns_sql.append(f"PRIMARY KEY ({pk_cols})")

    columns = ",\n  ".join(columns_sql)
    return f"CREATE TABLE {table.name} (\n  {columns}\n);"
```

#### Add Column
```python
def generate_add_column_sql(table: str, column: Column, dialect: str) -> str:
    sql = f"ALTER TABLE {table} ADD COLUMN {column.name} {column.data_type}"

    if not column.nullable:
        if column.default:
            sql += f" NOT NULL DEFAULT {column.default}"
        else:
            # BREAKING: Can't add NOT NULL without default
            raise IncompatibleChangeError(
                f"Cannot add NOT NULL column '{column.name}' without default value"
            )
    return sql + ";"
```

#### Modify Column
```python
def generate_modify_column_sql(table: str, old_col: Column, new_col: Column, dialect: str) -> str:
    if dialect == "postgresql":
        # PostgreSQL uses ALTER COLUMN syntax
        if old_col.data_type != new_col.data_type:
            return f"ALTER TABLE {table} ALTER COLUMN {new_col.name} TYPE {new_col.data_type};"
    elif dialect == "mysql":
        # MySQL uses MODIFY COLUMN syntax
        col_def = f"{new_col.name} {new_col.data_type}"
        if not new_col.nullable:
            col_def += " NOT NULL"
        return f"ALTER TABLE {table} MODIFY COLUMN {col_def};"
```

#### Drop Column
```python
def generate_drop_column_sql(table: str, column: str, dialect: str) -> str:
    return f"ALTER TABLE {table} DROP COLUMN {column};"
```

### 4.2 Statement Ordering (Dependency Resolution)

**Problem:** Foreign keys must be created AFTER referenced tables exist

**Algorithm:** Topological sort based on dependencies

```python
def order_statements(statements: List[SQLStatement]) -> List[SQLStatement]:
    # Build dependency graph
    graph = DependencyGraph()

    for stmt in statements:
        graph.add_node(stmt.sql, dependencies=stmt.dependencies)

    # Topological sort
    ordered = graph.topological_sort()

    return ordered
```

**Example:**
```sql
-- Original order (incorrect)
CREATE TABLE orders (
    id INT PRIMARY KEY,
    user_id INT,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
CREATE TABLE users (id INT PRIMARY KEY);

-- Correct order (after sorting)
CREATE TABLE users (id INT PRIMARY KEY);
CREATE TABLE orders (
    id INT PRIMARY KEY,
    user_id INT,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

### 4.3 Rollback Script Generation

**Strategy:** Invert every change

**Rules:**
- ADD TABLE → DROP TABLE
- DROP TABLE → CREATE TABLE (with original schema)
- ADD COLUMN → DROP COLUMN
- DROP COLUMN → ADD COLUMN (with original schema)
- MODIFY COLUMN → MODIFY COLUMN (revert to old type)

**Implementation:**
```python
def generate_rollback(diff: SchemaDiff, dialect: str) -> MigrationScript:
    rollback_statements = []

    for change in reversed(diff.changes):  # Reverse order
        match change.change_type:
            case "ADD_TABLE":
                rollback_statements.append(
                    SQLStatement(
                        sql=f"DROP TABLE {change.new_value.name};",
                        operation="DROP_TABLE"
                    )
                )
            case "DROP_TABLE":
                rollback_statements.append(
                    SQLStatement(
                        sql=generate_add_table_sql(change.old_value, dialect),
                        operation="CREATE_TABLE"
                    )
                )
            case "ADD_COLUMN":
                rollback_statements.append(
                    SQLStatement(
                        sql=generate_drop_column_sql(change.table_name, change.new_value.name, dialect),
                        operation="DROP_COLUMN"
                    )
                )
            # ... more cases

    return MigrationScript(
        migration_id=f"rollback_{diff.new_schema_version}",
        from_version=diff.new_schema_version,
        to_version=diff.old_schema_version,
        statements=rollback_statements,
        dialect=dialect
    )
```

---

## 5. COMPATIBILITY CHECKING

### 5.1 Type Compatibility Matrix

**Safe Type Changes (Widening):**
```python
SAFE_TYPE_WIDENING = {
    "TINYINT": ["SMALLINT", "INT", "BIGINT"],
    "SMALLINT": ["INT", "BIGINT"],
    "INT": ["BIGINT"],
    "FLOAT": ["DOUBLE"],
    "VARCHAR(N)": ["VARCHAR(M)"],  # where M > N
    "DECIMAL(P1,S1)": ["DECIMAL(P2,S2)"],  # where P2 > P1, S2 >= S1
}
```

**Breaking Type Changes (Narrowing or Incompatible):**
```python
BREAKING_TYPE_CHANGES = {
    "BIGINT": ["INT", "SMALLINT", "TINYINT"],  # May truncate
    "VARCHAR(M)": ["VARCHAR(N)"],  # where N < M - may truncate
    "DOUBLE": ["FLOAT"],  # Loss of precision
    "VARCHAR": ["INT"],  # Incompatible types
    "TEXT": ["VARCHAR(255)"],  # Potential truncation
}
```

**Implementation:**
```python
def is_type_compatible(old_type: str, new_type: str) -> bool:
    """Check if type change is safe (widening) or breaking (narrowing)"""
    if old_type == new_type:
        return True

    # Check widening rules
    if old_type in SAFE_TYPE_WIDENING:
        return new_type in SAFE_TYPE_WIDENING[old_type]

    # Special case: VARCHAR length increase
    if old_type.startswith("VARCHAR") and new_type.startswith("VARCHAR"):
        old_len = extract_varchar_length(old_type)
        new_len = extract_varchar_length(new_type)
        return new_len >= old_len

    # Default: incompatible
    return False
```

### 5.2 Constraint Compatibility

#### NOT NULL Constraint
**Challenge:** Adding NOT NULL to existing column with null values will fail

**Solution:** Pre-flight check
```python
def check_not_null_safety(table: str, column: str, connection: DatabaseConnection) -> bool:
    """Check if column contains null values"""
    result = connection.execute(
        f"SELECT COUNT(*) FROM {table} WHERE {column} IS NULL"
    )
    null_count = result.fetchone()[0]

    if null_count > 0:
        raise IncompatibleChangeError(
            f"Cannot add NOT NULL to {table}.{column}: {null_count} rows contain NULL"
        )
    return True
```

#### UNIQUE Constraint
**Challenge:** Adding UNIQUE constraint fails if duplicates exist

**Solution:** Pre-flight check + data migration
```python
def check_unique_safety(table: str, columns: List[str], connection: DatabaseConnection) -> bool:
    """Check if column(s) have duplicate values"""
    cols = ", ".join(columns)
    result = connection.execute(f"""
        SELECT {cols}, COUNT(*) as cnt
        FROM {table}
        GROUP BY {cols}
        HAVING cnt > 1
    """)

    duplicates = result.fetchall()
    if duplicates:
        raise IncompatibleChangeError(
            f"Cannot add UNIQUE constraint on {table}({cols}): {len(duplicates)} duplicate groups"
        )
    return True
```

### 5.3 Data Migration for Type Changes

**Example:** INT → VARCHAR conversion

```python
def generate_data_migration(change: SchemaChange) -> Optional[SQLStatement]:
    """Generate data transformation SQL for type changes"""
    if change.change_type == "MODIFY_COLUMN":
        old_col = change.old_value
        new_col = change.new_value

        if old_col.data_type == "INT" and new_col.data_type.startswith("VARCHAR"):
            # Safe conversion: INT → VARCHAR
            return SQLStatement(
                sql=f"UPDATE {change.table_name} SET {new_col.name} = CAST({old_col.name} AS VARCHAR);",
                operation="DATA_MIGRATION"
            )
        elif old_col.data_type.startswith("VARCHAR") and new_col.data_type == "INT":
            # Breaking conversion: VARCHAR → INT (may fail)
            return SQLStatement(
                sql=f"""
                    UPDATE {change.table_name}
                    SET {new_col.name} = CAST({old_col.name} AS INT)
                    WHERE {old_col.name} ~ '^[0-9]+$';  -- Only numeric strings
                """,
                operation="DATA_MIGRATION"
            )
    return None
```

---

## 6. ROLLBACK STRATEGIES

### 6.1 Schema Rollback

**Forward Migration:**
```sql
ALTER TABLE users ADD COLUMN email_verified BOOLEAN DEFAULT FALSE;
```

**Rollback Migration:**
```sql
ALTER TABLE users DROP COLUMN email_verified;
```

**Challenge:** What if user data was created in new column?

**Solution:** Rollback script preserves data structure but may lose data

### 6.2 Data Preservation Strategies

#### Strategy 1: Shadow Columns
**Approach:** Keep old column during migration, drop after success

```sql
-- Migration
ALTER TABLE users RENAME COLUMN price TO price_old;
ALTER TABLE users ADD COLUMN price DECIMAL(10,2);
UPDATE users SET price = CAST(price_old AS DECIMAL(10,2));
-- Keep price_old for rollback

-- Rollback
ALTER TABLE users DROP COLUMN price;
ALTER TABLE users RENAME COLUMN price_old TO price;
```

#### Strategy 2: Backup Tables
**Approach:** Copy data to backup table before destructive change

```sql
-- Migration
CREATE TABLE users_backup AS SELECT * FROM users;
ALTER TABLE users DROP COLUMN legacy_flags;

-- Rollback
ALTER TABLE users ADD COLUMN legacy_flags TEXT;
UPDATE users SET legacy_flags = (SELECT legacy_flags FROM users_backup WHERE users.id = users_backup.id);
DROP TABLE users_backup;
```

#### Strategy 3: Append-Only (Event Sourcing)
**Approach:** Never delete, only add with tombstone markers

```sql
-- Instead of DROP COLUMN, mark as deprecated
ALTER TABLE users ADD COLUMN legacy_flags_deprecated BOOLEAN DEFAULT TRUE;

-- Application ignores deprecated columns
```

### 6.3 Rollback Validation

**Test Rollback Scripts:**
```python
def test_migration_roundtrip():
    """Verify forward + rollback returns to original state"""
    # Snapshot original schema
    original_schema = schema_parser.parse_database(connection)

    # Apply migration
    executor.execute_migration(forward_migration, connection)
    migrated_schema = schema_parser.parse_database(connection)

    # Apply rollback
    executor.execute_migration(rollback_migration, connection)
    final_schema = schema_parser.parse_database(connection)

    # Schemas should match (modulo metadata)
    assert schemas_equivalent(original_schema, final_schema)
```

---

## 7. MULTI-DATABASE SUPPORT

### 7.1 Dialect Abstraction

**Strategy:** Abstract SQL generation behind dialect interface

```python
class SQLDialect(ABC):
    @abstractmethod
    def add_column_sql(self, table: str, column: Column) -> str:
        pass

    @abstractmethod
    def modify_column_sql(self, table: str, old_col: Column, new_col: Column) -> str:
        pass

    # ... more abstract methods

class PostgreSQLDialect(SQLDialect):
    def add_column_sql(self, table: str, column: Column) -> str:
        return f"ALTER TABLE {table} ADD COLUMN {column.name} {column.data_type};"

    def modify_column_sql(self, table: str, old_col: Column, new_col: Column) -> str:
        return f"ALTER TABLE {table} ALTER COLUMN {new_col.name} TYPE {new_col.data_type};"

class MySQLDialect(SQLDialect):
    def modify_column_sql(self, table: str, old_col: Column, new_col: Column) -> str:
        return f"ALTER TABLE {table} MODIFY COLUMN {new_col.name} {new_col.data_type};"
```

### 7.2 Database-Specific Features

**PostgreSQL-Specific:**
- `SERIAL` / `BIGSERIAL` auto-increment types
- Array columns: `INT[]`
- JSONB columns
- Partial indexes: `CREATE INDEX ... WHERE condition`

**MySQL-Specific:**
- `AUTO_INCREMENT` keyword
- `ENUM` and `SET` types
- Storage engines: `ENGINE=InnoDB`

**SQLite-Specific:**
- Limited ALTER TABLE support (can't drop columns in old versions)
- Dynamic typing (type affinity vs strict types)

**Implementation:**
```python
def normalize_type(raw_type: str, source_dialect: str) -> str:
    """Convert dialect-specific type to canonical form"""
    if source_dialect == "postgresql":
        if raw_type == "SERIAL":
            return "INTEGER AUTO_INCREMENT"
        elif raw_type == "JSONB":
            return "JSON"  # Generic JSON type
    elif source_dialect == "mysql":
        if raw_type == "INT AUTO_INCREMENT":
            return "INTEGER AUTO_INCREMENT"
    return raw_type
```

---

## 8. TEST SCENARIOS

### 8.1 Basic Diff Scenarios

#### Scenario 1: Add Table
**Setup:**
- Old schema: `users` table only
- New schema: `users` + `orders` table

**Expected Behavior:**
```python
diff = diff_engine.compute_diff(old_schema, new_schema)

assert len(diff.table_diff.added_tables) == 1
assert diff.table_diff.added_tables[0].name == "orders"
assert len(diff.changes) == 1
assert diff.changes[0].change_type == "ADD_TABLE"
assert diff.changes[0].change_impact == "SAFE"
```

#### Scenario 2: Drop Column
**Setup:**
- Old schema: `users(id, name, email, phone)`
- New schema: `users(id, name, email)`

**Expected Behavior:**
```python
diff = diff_engine.compute_diff(old_schema, new_schema)

column_diff = diff.column_diffs[0]
assert column_diff.table_name == "users"
assert len(column_diff.dropped_columns) == 1
assert column_diff.dropped_columns[0].name == "phone"

change = diff.changes[0]
assert change.change_type == "DROP_COLUMN"
assert change.change_impact == "DESTRUCTIVE"
```

#### Scenario 3: Modify Column Type
**Setup:**
- Old schema: `products(id, price INT)`
- New schema: `products(id, price DECIMAL(10,2))`

**Expected Behavior:**
```python
diff = diff_engine.compute_diff(old_schema, new_schema)

column_diff = diff.column_diffs[0]
assert len(column_diff.modified_columns) == 1
old_col, new_col = column_diff.modified_columns[0]
assert old_col.data_type == "INT"
assert new_col.data_type == "DECIMAL"

change = diff.changes[0]
assert change.change_type == "MODIFY_COLUMN"
assert change.change_impact == "BREAKING"  # Needs data migration
```

### 8.2 Migration Generation Scenarios

#### Scenario 4: Generate Add Column Migration
**Setup:**
- Diff: Add `email_verified BOOLEAN DEFAULT FALSE` to `users`

**Expected Behavior:**
```python
migration = script_generator.generate_migration(diff, dialect="postgresql")

assert len(migration.statements) == 1
assert "ALTER TABLE users ADD COLUMN email_verified BOOLEAN DEFAULT FALSE" in migration.statements[0].sql
assert migration.breaking_changes == []  # Safe change
```

#### Scenario 5: Generate Rollback Script
**Setup:**
- Forward migration: Add column `email_verified`

**Expected Behavior:**
```python
rollback = script_generator.generate_rollback(diff, dialect="postgresql")

assert len(rollback.statements) == 1
assert "ALTER TABLE users DROP COLUMN email_verified" in rollback.statements[0].sql
```

#### Scenario 6: Statement Ordering
**Setup:**
- Add `orders` table with FK to `users`
- Add `users` table

**Expected Behavior:**
```python
migration = script_generator.generate_migration(diff, dialect="postgresql")

# Users must be created before orders (FK dependency)
statements = migration.statements
assert statements[0].operation == "CREATE_TABLE"
assert "users" in statements[0].sql
assert statements[1].operation == "CREATE_TABLE"
assert "orders" in statements[1].sql
```

### 8.3 Compatibility Checking Scenarios

#### Scenario 7: Safe Type Widening
**Setup:**
- Change `age INT` → `age BIGINT`

**Expected Behavior:**
```python
impact = compatibility_checker.classify_change(change)

assert impact.impact_level == "SAFE"
assert impact.requires_data_migration == False
assert impact.data_loss_risk == False
```

#### Scenario 8: Breaking Type Narrowing
**Setup:**
- Change `description TEXT` → `description VARCHAR(255)`

**Expected Behavior:**
```python
impact = compatibility_checker.classify_change(change)

assert impact.impact_level == "BREAKING"
assert impact.data_loss_risk == True  # May truncate long text
```

#### Scenario 9: NOT NULL Safety Check
**Setup:**
- Add NOT NULL to column with existing null values

**Expected Behavior:**
```python
with pytest.raises(IncompatibleChangeError) as exc:
    compatibility_checker.check_not_null_safety("users", "phone", connection)

assert "42 rows contain NULL" in str(exc.value)
```

### 8.4 Execution Scenarios

#### Scenario 10: Successful Migration
**Setup:**
- Valid migration script
- Database in expected state

**Expected Behavior:**
```python
result = executor.execute_migration(migration, connection)

assert result.success == True
assert result.executed_statements == len(migration.statements)
assert result.error_message is None
```

#### Scenario 11: Migration Failure and Rollback
**Setup:**
- Migration with invalid SQL (syntax error)

**Expected Behavior:**
```python
result = executor.execute_migration(migration, connection)

assert result.success == False
assert result.failed_statement is not None
assert "syntax error" in result.error_message.lower()

# Database rolled back to original state (transaction rollback)
schema_after_failure = schema_parser.parse_database(connection)
assert schemas_equivalent(schema_before, schema_after_failure)
```

#### Scenario 12: Dry Run Validation
**Setup:**
- Migration script with breaking changes

**Expected Behavior:**
```python
report = executor.dry_run_migration(migration, connection)

assert report.valid == True  # Syntactically valid
assert len(report.warnings) > 0  # Breaking changes detected
assert "BREAKING" in report.warnings[0]
# Database unchanged
```

### 8.5 Migration History Scenarios

#### Scenario 13: Track Migration History
**Setup:**
- Apply migration v1 → v2

**Expected Behavior:**
```python
executor.execute_migration(migration_v1_to_v2, connection)

history = history_tracker.get_migration_history()
assert len(history) == 1
assert history[0].migration_id == "v1_to_v2"
assert history[0].success == True
```

#### Scenario 14: Prevent Re-Application
**Setup:**
- Migration v1 → v2 already applied
- Attempt to apply again

**Expected Behavior:**
```python
assert history_tracker.is_migration_applied("v1_to_v2") == True

with pytest.raises(MigrationAlreadyAppliedError):
    executor.execute_migration(migration_v1_to_v2, connection)
```

#### Scenario 15: Compute Migration Path
**Setup:**
- Current version: v1
- Target version: v4
- Available migrations: v1→v2, v2→v3, v3→v4

**Expected Behavior:**
```python
path = history_tracker.compute_migration_path("v1", "v4")

assert path.migration_chain == ["v1_to_v2", "v2_to_v3", "v3_to_v4"]
assert path.total_steps == 3
```

---

## 9. TESTING STRATEGY

### 9.1 Unit Tests

**SchemaParser (10 tests):**
- `test_parse_database_extracts_tables`
- `test_parse_database_extracts_columns`
- `test_parse_database_extracts_indexes`
- `test_parse_database_extracts_foreign_keys`
- `test_parse_ddl_file_creates_schema`
- `test_normalize_schema_canonical_types`
- `test_extract_constraints_accurate`
- `test_handle_vendor_specific_syntax`
- `test_parse_empty_database`
- `test_parse_large_schema_performance`

**DiffEngine (15 tests):**
- `test_diff_detects_added_table`
- `test_diff_detects_dropped_table`
- `test_diff_detects_modified_table`
- `test_diff_detects_added_column`
- `test_diff_detects_dropped_column`
- `test_diff_detects_modified_column`
- `test_diff_detects_column_rename`
- `test_diff_detects_index_changes`
- `test_diff_detects_foreign_key_changes`
- `test_diff_empty_when_schemas_identical`
- `test_diff_classifies_breaking_changes`
- `test_diff_classifies_safe_changes`
- `test_diff_minimal_changeset`
- `test_diff_large_schema_performance`
- `test_diff_ignores_metadata_changes`

**ScriptGenerator (12 tests):**
- `test_generate_add_table_sql`
- `test_generate_drop_table_sql`
- `test_generate_add_column_sql`
- `test_generate_drop_column_sql`
- `test_generate_modify_column_sql`
- `test_generate_rollback_script`
- `test_order_statements_by_dependencies`
- `test_dialect_specific_syntax_postgresql`
- `test_dialect_specific_syntax_mysql`
- `test_dialect_specific_syntax_sqlite`
- `test_breaking_change_warnings`
- `test_estimate_migration_duration`

**CompatibilityChecker (12 tests):**
- `test_classify_safe_type_widening`
- `test_classify_breaking_type_narrowing`
- `test_classify_destructive_drop_column`
- `test_type_compatibility_int_to_bigint`
- `test_type_incompatibility_varchar_to_int`
- `test_check_not_null_safety_no_nulls`
- `test_check_not_null_safety_has_nulls`
- `test_check_unique_safety_no_duplicates`
- `test_check_unique_safety_has_duplicates`
- `test_generate_data_migration_type_change`
- `test_validate_foreign_key_references`
- `test_detect_orphaned_fk_references`

**MigrationExecutor (10 tests):**
- `test_execute_migration_success`
- `test_execute_migration_failure_rollback`
- `test_dry_run_validation`
- `test_transaction_atomicity`
- `test_record_migration_history`
- `test_get_migration_history`
- `test_execute_migration_timeout`
- `test_validate_schema_after_migration`
- `test_concurrent_migration_blocking`
- `test_migration_progress_reporting`

**HistoryTracker (8 tests):**
- `test_initialize_history_table`
- `test_is_migration_applied_true`
- `test_is_migration_applied_false`
- `test_compute_migration_path_single_step`
- `test_compute_migration_path_multi_step`
- `test_compute_migration_path_no_path`
- `test_detect_conflicts_divergent_branches`
- `test_migration_history_persistence`

### 9.2 Integration Tests

**Schema Migration Integration (12 tests):**
- `test_full_migration_workflow_add_table`
- `test_full_migration_workflow_modify_column`
- `test_full_migration_workflow_drop_column`
- `test_rollback_after_migration`
- `test_migration_chain_v1_to_v3`
- `test_migration_with_data_preservation`
- `test_breaking_change_validation_fails`
- `test_multi_database_same_migration`
- `test_concurrent_schema_reads_during_migration`
- `test_large_table_migration_performance`
- `test_foreign_key_dependency_ordering`
- `test_index_creation_on_large_table`

### 9.3 Property-Based Tests (Hypothesis)

**Diff Idempotence:**
```python
@given(st.builds(Schema))
def test_diff_self_is_empty(schema):
    """Diffing schema with itself produces no changes"""
    diff = diff_engine.compute_diff(schema, schema)
    assert len(diff.changes) == 0
```

**Rollback Roundtrip:**
```python
@given(st.builds(SchemaDiff))
def test_migration_rollback_roundtrip(diff):
    """Forward + rollback returns to original state"""
    forward = script_generator.generate_migration(diff, "postgresql")
    rollback = script_generator.generate_rollback(diff, "postgresql")

    # Apply forward
    executor.execute_migration(forward, test_db)
    schema_after_forward = schema_parser.parse_database(test_db)

    # Apply rollback
    executor.execute_migration(rollback, test_db)
    schema_after_rollback = schema_parser.parse_database(test_db)

    # Should match original
    assert schemas_equivalent(original_schema, schema_after_rollback)
```

---

## 10. PARAGON-SPECIFIC INTEGRATION

### 10.1 Graph Database Integration

**Map Schema Migration to ParagonDB:**
- Schema versions → `NodeType.VERSION`
- Migrations → `NodeType.MIGRATION`
- Tables → `NodeType.ENTITY`
- Columns → `NodeType.ATTRIBUTE`
- Schema changes → `EdgeType.MIGRATES_TO`
- Dependencies → `EdgeType.DEPENDS_ON` (FK relationships)

**Benefits:**
- Visualize migration history as DAG
- Compute migration paths using `get_waves()`
- Detect circular dependencies in schema
- Teleology: trace schema evolution over time

### 10.2 Orchestrator Integration

**TDD Cycle Mapping:**
1. **DIALECTIC:** Clarify breaking vs safe changes
2. **RESEARCH:** Study database migration best practices (Flyway, Liquibase)
3. **PLAN:** Architect creates component breakdown
4. **BUILD:** Builder implements DiffEngine, ScriptGenerator, etc.
5. **TEST:** Tester verifies diff correctness, rollback safety

**Checkpoint Integration:**
- Store migration history in graph_db
- Track applied migrations as completed nodes

### 10.3 Rerun Visualization

**Visualizations:**
- Schema evolution timeline (versions as nodes)
- Migration dependency graph (foreign key relationships)
- Breaking change heatmap (destructive operations)
- Rollback simulation (show state reversion)
- Diff visualization (before/after schema comparison)

---

## 11. EXTENSION POINTS

### 11.1 Advanced Features (Out of Scope for MVP)

**Zero-Downtime Migrations:**
- Blue-green deployment schema strategy
- Shadow tables for gradual cutover
- Dual-write during migration

**Schema Branching:**
- Support multiple schema versions in production
- Feature flags for schema changes
- Merge divergent schema branches

**Declarative Migrations:**
- Define target schema, auto-generate migrations
- Compare with Terraform/Pulumi infrastructure-as-code model

### 11.2 Integration with ORMs

**SQLAlchemy Integration:**
- Generate migrations from SQLAlchemy models
- Compare model definitions with database schema
- Auto-detect model changes

**Django Integration:**
- Similar to Django migrations system
- Generate migration files from models.py changes

---

## 12. EVALUATION CRITERIA

### 12.1 Orchestrator Evaluation

**How well does the orchestrator handle this problem?**
- Does it recognize the graph-based nature (dependency ordering)?
- Does it design diff algorithm correctly?
- Does it implement safety checks (breaking changes)?
- Does it handle rollbacks properly?
- Does it support multiple database dialects?

### 12.2 Code Quality Metrics

**Generated Code Quality:**
- Test coverage ≥ 90%
- Correct SQL generation (syntactically valid)
- Safe migrations (no data loss)
- Schema validation with msgspec
- Proper error handling

### 12.3 Performance Validation

**Benchmark Targets:**
```python
# Protocol Alpha extension
def test_schema_migration_performance():
    """Test schema migration engine performance"""
    # Diff large schema
    assert diff_time_1000_tables < 1.0  # 1 second

    # Generate migration
    assert script_gen_time < 0.5  # 500ms

    # Execute migration (small table)
    assert migration_time_1000_rows < 2.0  # 2 seconds
```

---

## 13. IMPLEMENTATION CHECKLIST

### Phase 1: Foundation (Day 1-2)
- [ ] Define all msgspec schemas (Column, Table, Index, etc.)
- [ ] Implement SchemaParser for PostgreSQL
- [ ] Implement basic DiffEngine (table and column diffs)
- [ ] Unit tests for SchemaParser and DiffEngine (20 tests)

### Phase 2: Migration Generation (Day 2-3)
- [ ] Implement ScriptGenerator with SQL generation
- [ ] Implement statement ordering (dependency resolution)
- [ ] Implement rollback script generation
- [ ] Unit tests (12 tests)
- [ ] Integration test: generate and validate migration script

### Phase 3: Compatibility (Day 3-4)
- [ ] Implement CompatibilityChecker
- [ ] Implement type compatibility matrix
- [ ] Implement NOT NULL and UNIQUE safety checks
- [ ] Implement data migration script generation
- [ ] Unit tests (12 tests)

### Phase 4: Execution (Day 4-5)
- [ ] Implement MigrationExecutor with transaction support
- [ ] Implement HistoryTracker with migration logging
- [ ] Implement dry-run mode
- [ ] Unit tests (18 tests)
- [ ] Integration test: execute migration on test database

### Phase 5: Multi-Database (Day 5-6)
- [ ] Add MySQL dialect support
- [ ] Add SQLite dialect support
- [ ] Implement dialect abstraction layer
- [ ] Cross-database integration tests
- [ ] Property-based tests (Hypothesis)
- [ ] Performance benchmarks
- [ ] Documentation and examples

---

## 14. EXPECTED LEARNING OUTCOMES

### For the Orchestrator:
1. **Graph algorithms:** Dependency resolution via topological sort
2. **Diff algorithms:** Set-based comparison with heuristics
3. **SQL generation:** Database-specific syntax handling
4. **Safety analysis:** Breaking vs safe change classification

### For the Paragon System:
1. **Schema evolution:** Modeling database changes in graph
2. **Backward compatibility:** Ensuring safe rollback paths
3. **Multi-dialect support:** Abstraction over database differences
4. **Data safety:** Preventing data loss in migrations

---

## 15. DELIVERABLES

### Code Artifacts:
- `schema_migration/` module with 6 Python files
- `tests/schema_migration/` with 75+ unit tests
- `tests/integration/test_schema_migration.py` with 12 integration tests
- Example migrations in `examples/schema_migration_demo/`

### Documentation:
- API documentation (docstrings)
- User guide (how to create and apply migrations)
- Architecture diagram (component interactions)
- Migration safety guide (breaking changes checklist)

### Metrics:
- Test coverage report (≥90% target)
- Performance benchmark results
- Compatibility matrix (type conversions)

---

## APPENDIX A: Reference Implementations

### Real-World Schema Migration Tools:
1. **Flyway:** Java-based migration tool with versioned SQL scripts
2. **Liquibase:** Database-agnostic migration with XML/YAML/JSON definitions
3. **Alembic:** Python migration tool for SQLAlchemy
4. **Django Migrations:** Built-in migration system for Django ORM
5. **gh-ost:** GitHub's triggerless online schema migration for MySQL

### Key Learnings:
- Flyway's versioned migration approach maps to our history tracking
- Liquibase's changeset model is similar to our SchemaDiff
- Alembic's auto-generate feature is similar to our diff engine

---

## APPENDIX B: Prompt Template for Orchestrator

**When submitting this problem to Paragon:**

```markdown
I need a schema migration engine for my database applications.

REQUIREMENTS:
1. Compare two database schemas and generate a diff
2. Create SQL migration scripts to apply changes
3. Detect breaking changes (drop column, change type)
4. Generate rollback scripts for every migration
5. Execute migrations safely with transactions
6. Track migration history to prevent re-application
7. Support PostgreSQL, MySQL, and SQLite

EXAMPLE WORKFLOW:
- Current schema: users(id, name, email)
- New schema: users(id, name, email, email_verified BOOLEAN)
- System generates: ALTER TABLE users ADD COLUMN email_verified BOOLEAN DEFAULT FALSE;
- System generates rollback: ALTER TABLE users DROP COLUMN email_verified;
- System warns if breaking changes detected (e.g., DROP COLUMN phone)

CONSTRAINTS:
- Must handle 1000-table schemas efficiently
- Must prevent data loss in migrations
- Must support type conversions with data migration
- Must order statements by dependencies (foreign keys)
- Must validate constraints before applying changes

Please implement this system using the Paragon TDD workflow.
```

**Expected Orchestrator Behavior:**
1. DIALECTIC identifies compatibility questions (which databases?)
2. RESEARCH investigates migration patterns (Flyway, Alembic)
3. ARCHITECT creates component breakdown (matches Section 2.1)
4. BUILDER generates code with msgspec schemas
5. TESTER verifies all scenarios from Section 8

---

**END OF RESEARCH DOCUMENT**
