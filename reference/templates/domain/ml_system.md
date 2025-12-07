# Machine Learning System Domain Template

This template extends the generic templates with ML system-specific questions.

## ML Problem Type
**Questions to ask:**
- What type of ML problem (classification, regression, clustering, ranking, generation)?
- Is this supervised, unsupervised, or reinforcement learning?
- What is the input and output of the model?
- What is the business metric you're optimizing for?
- What is the baseline to beat (heuristic, existing model)?

**Example answers:**
> - Problem: Multi-class classification (predict service conflict severity)
> - Learning: Supervised (labeled historical conflicts)
> - Input: Dependency graph features (graph structure, version constraints)
> - Output: Conflict severity (HIGH, MEDIUM, LOW, NONE)
> - Metric: F1 score (balance precision/recall for alerts)
> - Baseline: Rule-based heuristic (50% F1)

---

## Data Collection & Labeling
**Questions to ask:**
- What data sources will you use for training?
- How much labeled data is available (sample size)?
- What is the data labeling process (manual, programmatic, semi-supervised)?
- What is the data quality (noise, errors, missing values)?
- How will you handle class imbalance?
- Will you collect data continuously or one-time?

**Example answers:**
> - Sources: Historical deployment logs, service metadata, incident reports
> - Size: 10,000 labeled examples (past 2 years of conflicts)
> - Labeling: Programmatic from incident severity + manual review for edge cases
> - Quality: ~5% mislabeled (verified via sample audit)
> - Imbalance: 70% NONE, 20% LOW, 8% MEDIUM, 2% HIGH (use class weights)
> - Collection: Continuous (label new conflicts as they occur)

---

## Feature Engineering
**Questions to ask:**
- What features will you extract from raw data?
- Will you use hand-crafted features or learned representations?
- How will you handle categorical features (one-hot, embeddings)?
- How will you handle numerical features (scaling, binning)?
- Will you use feature crosses or polynomial features?
- How will you handle missing features?
- Will you do feature selection?

**Example answers:**
> - Features:
>   - Graph structure: In-degree, out-degree, PageRank
>   - Version constraints: Semver compatibility scores
>   - Historical: Past conflict rate for service pair
>   - Temporal: Days since last deployment
> - Type: Hand-crafted (domain knowledge-driven)
> - Categorical: One-hot encoding for service types
> - Numerical: StandardScaler for all numeric features
> - Crosses: Service-pair interactions
> - Missing: Fill with median for numeric, "UNKNOWN" for categorical
> - Selection: Use Lasso for feature selection (reduce to top 20)

---

## Model Selection
**Questions to ask:**
- What model architecture(s) will you try (linear, tree-based, neural network)?
- Why is this architecture appropriate for the problem?
- Will you ensemble multiple models?
- What is the model complexity budget (interpretability vs accuracy)?
- Will you use pretrained models or train from scratch?
- What libraries/frameworks (scikit-learn, XGBoost, PyTorch)?

**Example answers:**
> - Architecture: Gradient Boosted Trees (XGBoost)
> - Rationale: Works well with tabular data, handles feature interactions
> - Ensemble: Compare XGBoost vs Random Forest, ensemble if beneficial
> - Complexity: Prefer interpretable (need to explain predictions)
> - Pretrained: Not applicable (custom domain, train from scratch)
> - Framework: XGBoost for model, scikit-learn for preprocessing

---

## Training & Optimization
**Questions to ask:**
- What is the training procedure (batch, online, incremental)?
- How will you split data (train/val/test)?
- What loss function will you use?
- What optimization algorithm (SGD, Adam)?
- How will you tune hyperparameters (grid search, Bayesian)?
- What stopping criteria (early stopping, fixed epochs)?
- How long does training take?

**Example answers:**
> - Training: Batch (weekly retraining on new data)
> - Split: 70% train, 15% validation, 15% test (temporal split, not random)
> - Loss: Weighted cross-entropy (for class imbalance)
> - Optimizer: XGBoost's built-in (tree boosting)
> - Hyperparameters: Optuna for Bayesian optimization (100 trials)
>   - max_depth, learning_rate, n_estimators, subsample
> - Stopping: Early stopping (10 rounds without val improvement)
> - Duration: ~30 minutes for full hyperparameter search

---

## Evaluation & Metrics
**Questions to ask:**
- What metrics will you use to evaluate the model?
- What is the target performance (acceptable threshold)?
- How will you evaluate on different data slices (fairness)?
- Will you use offline metrics or online A/B tests?
- How will you validate model robustness (adversarial, distribution shift)?
- What is the metric for business impact?

**Example answers:**
> - Metrics:
>   - Primary: F1 score (macro-averaged across classes)
>   - Secondary: Precision/Recall per class, ROC-AUC
> - Target: F1 > 0.75 (beat baseline of 0.50)
> - Slices: Evaluate separately on high-traffic vs low-traffic services
> - Testing: Offline validation first, then canary rollout (10% traffic)
> - Robustness: Test on adversarial cases (deliberately conflicting configs)
> - Business: Reduction in deployment rollbacks (track in production)

---

## Model Deployment
**Questions to ask:**
- How will the model be deployed (API, batch, embedded)?
- What is the serving infrastructure (Flask, FastAPI, TensorFlow Serving)?
- What is the prediction latency requirement (real-time, batch)?
- How will you version models?
- How will you roll out new models (canary, blue/green)?
- What is the rollback strategy?

**Example answers:**
> - Deployment: REST API endpoint for real-time predictions
> - Infrastructure: FastAPI + uvicorn
> - Latency: < 100ms p95 (interactive use)
> - Versioning: Model registry (MLflow), semantic versioning
> - Rollout: Canary (10% → 50% → 100% over 3 days)
> - Rollback: Automatic if error rate > 5% or latency > 200ms

---

## Model Monitoring
**Questions to ask:**
- What will you monitor in production (accuracy, latency, errors)?
- How will you detect model degradation?
- How will you detect data drift or concept drift?
- What alerts will you configure?
- How will you collect feedback for retraining?
- What is the retraining cadence?

**Example answers:**
> - Monitoring:
>   - Model performance: Precision, recall (via labeled feedback)
>   - System: Prediction latency, error rate, throughput
>   - Data: Feature distribution drift (KL divergence)
> - Degradation: Alert if F1 drops > 10% from validation
> - Drift: Monitor input feature distributions weekly
> - Alerts: Slack for drift detection, PagerDuty for model errors
> - Feedback: Collect labels from incident reports (manual review)
> - Retraining: Weekly with new labeled data

---

## Feature Store & Data Pipeline
**Questions to ask:**
- Will you use a feature store (Feast, Tecton, custom)?
- How will you keep features consistent between training and serving?
- How will you handle feature staleness?
- What is the feature computation latency?
- How will you backfill features for training?

**Example answers:**
> - Feature store: Not needed for v1 (simple feature pipeline)
> - Consistency: Same feature computation code for train & serve
> - Staleness: Compute features on-demand (fresh for every prediction)
> - Latency: < 50ms to compute features
> - Backfill: Recompute features from historical dependency graph snapshots

---

## Model Interpretability
**Questions to ask:**
- Do you need to explain predictions (regulatory, user trust)?
- What interpretability techniques (SHAP, LIME, feature importance)?
- Will you provide global or local explanations?
- How will you present explanations to users?

**Example answers:**
> - Explainability: Yes, users need to understand why conflicts are predicted
> - Techniques: SHAP values for individual predictions
> - Scope: Local explanations (per prediction)
> - Presentation: API returns top 3 features + their contribution
>   - Example: "Conflict predicted due to: version incompatibility (0.4), high dependency depth (0.3), recent deployment (0.2)"

---

## Bias & Fairness
**Questions to ask:**
- Are there fairness concerns (protected attributes, disparate impact)?
- How will you measure bias (demographic parity, equal opportunity)?
- What mitigation strategies will you use (reweighting, threshold tuning)?
- How will you audit the model for bias?

**Example answers:**
> - Fairness: Ensure model doesn't favor high-traffic services over low-traffic
> - Measure: Compare false positive rate across service tiers
> - Mitigation: Balanced sampling during training (equal examples per tier)
> - Audit: Quarterly fairness review on stratified test sets

---

## Privacy & Security
**Questions to ask:**
- Does the model process PII or sensitive data?
- How will you protect training data (encryption, access control)?
- Will you use privacy-preserving techniques (differential privacy, federated learning)?
- How will you prevent model inversion or membership inference attacks?
- What are the compliance requirements (GDPR, HIPAA)?

**Example answers:**
> - PII: No PII (only technical service metadata)
> - Protection: Training data encrypted at rest, access-controlled
> - Privacy techniques: Not needed (no sensitive data)
> - Attacks: Model doesn't expose training data (low risk)
> - Compliance: SOC2 (audit logging for model updates)

---

## Experimentation & A/B Testing
**Questions to ask:**
- How will you run experiments (A/B tests, multi-armed bandits)?
- What is the experiment unit (user, session, request)?
- What is the minimum sample size for statistical significance?
- How long will experiments run?
- How will you handle multiple concurrent experiments?

**Example answers:**
> - Experiments: A/B test (control vs new model)
> - Unit: Per deployment (service deployment is the unit)
> - Sample size: 1,000 deployments per variant (power analysis)
> - Duration: 2 weeks (capture weekly deployment cycle)
> - Concurrent: Mutually exclusive buckets (no interference)

---

## Model Versioning & Governance
**Questions to ask:**
- How will you version models and training data?
- How will you track model lineage (data → features → model)?
- What is the model approval process?
- How will you archive old models?
- What documentation is required for each model?

**Example answers:**
> - Versioning: Git for code, MLflow for models, DVC for data
> - Lineage: MLflow tracks data version, feature version, hyperparameters
> - Approval: Staging validation + manual review before production
> - Archival: Keep last 5 production models, archive older to S3
> - Documentation: Model card with metrics, data, assumptions, limitations

---

## Error Handling & Fallbacks
**Questions to ask:**
- What happens if the model fails to predict?
- Will you have a fallback model or heuristic?
- How will you handle out-of-distribution inputs?
- What is the degraded mode behavior?

**Example answers:**
> - Failure: Return fallback to rule-based heuristic
> - Fallback: Simple heuristic (version compatibility check only)
> - OOD: Detect using feature distribution, flag as low-confidence
> - Degraded: If model service down, use cached predictions (5min TTL) or fallback

---

## Cost & Resource Management
**Questions to ask:**
- What are the cost drivers (compute for training/serving, storage)?
- How will you optimize training costs?
- How will you optimize serving costs?
- What hardware will you use (CPU, GPU, TPU)?
- What is the budget for ML operations?

**Example answers:**
> - Costs: Training (AWS EC2), serving (containers), storage (S3)
> - Training optimization: Use spot instances, cache preprocessed features
> - Serving optimization: Batch predictions where possible, use CPU (no GPU needed)
> - Hardware: CPU only (XGBoost doesn't require GPU)
> - Budget: $200/month (mostly serving infrastructure)

---

## Testing Strategy
**Questions to ask:**
- How will you test the model (unit, integration, regression)?
- How will you test for model quality (smoke tests, invariance tests)?
- How will you test the serving infrastructure?
- Will you use synthetic data for testing?

**Example answers:**
> - Unit: Test preprocessing functions, feature engineering
> - Integration: End-to-end pipeline (raw data → prediction)
> - Regression: Track performance on fixed test set over time
> - Quality: Invariance tests (flipping irrelevant feature shouldn't change prediction)
> - Infrastructure: Load testing API (1000 req/s)
> - Synthetic: Generate edge cases (missing features, extreme values)
