PyO3 and zero copy

## 1. ⚠️ What is At Issue (The Core Conflict)

The issue is a conflict between **performance-driven architectural complexity (PyO3/Zero-Copy)** and **development velocity/flexibility**.

| Architectural Choice | Description | The Issue |
| :--- | :--- | :--- |
| **High-Performance (Zero-Copy)** | Using Rust's **PyO3** and Arrow's memory sharing to bypass Python object creation overhead and achieve maximum data ingestion speeds (10M+ nodes/sec). | It requires highly **rigid, manually-defined Rust bindings** that break whenever the Python schema changes, dramatically increasing maintenance cost and slowing the iteration speed of the **Paragon** system. |
| **High-Flexibility (Current `msgspec`)** | Using Python/C-accelerated serialization (`msgspec`) to ingest data efficiently. | This approach incurs a small performance overhead from the Python object creation tax (slower than zero-copy) but offers **maximum development flexibility**, allowing the schema to evolve rapidly without touching the Rust core. |

The conflict is: **Do we pay a massive development/maintenance tax now for a performance gain we don't currently need?** (Answer: No, stick with flexibility).

---

## 2. 🧱 Schema Maturity Requirement

To make the zero-copy approach *not so bad* to do, the schema needs to be **highly mature and functionally immutable**.

### Required Maturity Level: **Version Locked (95%+ Immature)**

The schema needs to be:

* **Versioned:** The input structure must be formally locked as "V1.0," with a strict deprecation path for V2.0.
* **Highly Rigid:** The probability of adding, removing, or changing the data type of a *critical* field must be **LOW** (ideally < 5%) over the next 12-18 months. This is because **every single schema change** would require:
    1.  Updating the Python source definition.
    2.  Updating the Rust struct definition (in `PyO3`).
    3.  Updating all Rust code that interacts with the new/changed field.
    4.  Recompiling the Rust binary.
    5.  Re-running CI/CD checks for the entire toolchain.

### Our Current State: **Low Rigidity (Actively Evolving)**

Given the recent additions of `SignatureChain`, `W3C PROV types`, and multiple new fields/enums, the schema is correctly identified as **actively evolving**. Implementing zero-copy now would significantly damage development velocity.

---

## 3. ⏱️ When and How It Should Be Done

The Zero-Copy infrastructure should be treated as an **"Escape Hatch"** for a time when the strategic trade-offs have shifted.

| Aspect | Guideline for Implementation |
| :--- | :--- |
| **When to Start** | **When the current `msgspec` solution hits a measurable, unacceptable performance ceiling.** This is defined as: **Benchmark data shows ingestion time for a critical workload is > 30 seconds (or a functionally defined limit) AND no other Python-side optimization can solve it.** |
| **How to Do It** | **As a new, parallel ingestion path.** Implement the zero-copy logic in a separate method (e.g., `paragon.ingest_nodes_arrow()`) alongside the existing `paragon.ingest_nodes_msgspec()`. This allows high-volume users to opt into the complexity without forcing the transition on the whole system. |
| **Pre-requisite** | The schema must have been stable and unchanged for **at least 6 months** prior to starting the engineering effort. |

---

## 4. 📝 Future Note Placement in a Repo

The most common and effective places to put a future architectural note (an "Escape Hatch" or "Phase L∞" option) are:

1.  **`ARCHITECTURE.md`:** This is the ideal location. A section titled "Performance Escape Hatches" or "Phase II/L∞: Zero-Copy Ingestion" provides necessary context for future architects and contributors.
2.  **Code Comments/TODOs:** Add a concise `// TODO(FuturePerf): Consider PyO3/Arrow Zero-Copy when load exceeds X nodes/sec.` right next to the current `msgspec` ingestion entry point. This serves as a direct reminder in the code itself.
3.  **The Project's Wiki or Documentation:** If the repository has a dedicated documentation portal, a **"Future Roadmap"** page is a great place to document the decision and the performance metrics required to trigger the change.

I recommend starting with **`ARCHITECTURE.md`** as it captures the strategic reasoning best.
