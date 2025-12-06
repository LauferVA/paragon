# SPEC-b08457ae: SHA256 hash function: accept bytes input, return hex string,...

## Metadata
- **Status**: VERIFIED
- **Created**: 2025-12-06T06:08:52.201275+00:00
- **Created By**: architect-agent
- **Version**: 1

## Specification
SHA256 hash function: accept bytes input, return hex string, handle errors gracefully

## Requirement
**REQ-8e6243b6**
Implement a cryptographic hash function module with SHA256 support

## Implementation
### CODE-97b726ae
**File**: `src/crypto/hash_utils.py`
```python
def hash_data(data: bytes) -> str:
    """Compute SHA256 hash of input data."""
    import hashlib
    if not isinstance(data, bytes):
        raise TypeError("Input must be bytes")
    return hashlib.sha256(data).hexdigest()

```
