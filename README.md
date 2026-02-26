

# AVC Edge SDK  
## CPU-Optimized Vehicle Classification from Raw IR-Curtain Signals

A deployment-focused redesign of a vehicle classification system for CPU-only edge devices in real-world toll plaza environments.

This system replaces an inefficient image-based modeling pipeline with direct feature modeling from raw IR-curtain binary signals. The redesigned architecture aligns with hardware constraints and delivers real-time performance without GPU dependency.

---

# 1. Problem Statement

The previous generation model converted raw IR sensor binary strings into profile images before classification:
```bash
Binary String → Image Reconstruction → CNN → Prediction
```
This architecture assumed GPU availability and imposed unnecessary computational overhead.

### Why the Image-Based Approach Failed on Edge Devices

The deployed edge devices operated on **CPU-only hardware**.

This created multiple issues:

- CNN inference was computationally expensive on CPU
- Image reconstruction added preprocessing latency
- Memory usage exceeded optimal limits
- End-to-end latency became unsuitable for real-time toll environments
- Error rates were inflated due to image interpolation artifacts

The system was architected for GPU-class compute but deployed on CPU-only hardware.

---

# 2. Dataset & Experimental Setup

The full dataset consists of **10 days of real toll plaza traffic data** collected from the IR-curtain system.

### Data Split

- 80% Training  
- 10% Validation  
- 10% Testing  

The **test dataset corresponds to a full operational day**, containing **6,583 samples**.

This evaluation reflects real-world traffic distribution, not an artificially balanced dataset.

Weighted metrics therefore approximate real deployment performance.

Datasets are not included in the repo due to size/sensitivity

---

# 3. System Redesign: CPU-Aligned Direct Feature Modeling

The redesigned pipeline operates directly on binary sensor data:
```bash
Binary String → Structured Feature Extraction → MLP → Prediction
```
Key design goals:

- Eliminate image reconstruction
- Remove GPU dependency
- Minimize CPU load
- Reduce memory footprint
- Preserve predictive performance

Instead of modeling artificial pixel structure, the system models structured signal statistics directly.

---

# 4. Feature Engineering

`feature_extraction.py` transforms each binary string into:

- An 80-sensor × N-slice structured matrix  
- A fixed 41-dimensional feature vector  

Properties:

- Deterministic transformation  
- Fixed dimensionality  
- No interpolation artifacts  
- Minimal memory overhead  
- CPU-efficient computation  

The feature representation aligns with the physical sensing process.

---

# 5. Architecture
```bash
Input (.txt binary)
↓
Feature Extraction (CPU)
↓
Standard Scaling
↓
MLP Inference (CPU-Optimized)
↓
Prediction + Latency
```
The architecture is explicitly optimized for CPU execution.

---

# 6. Technology Stack

- Python 3  
- NumPy  
- csv (built-in module)  
- Scikit-learn (MLPClassifier, StandardScaler, LabelEncoder)  
- Imbalanced-learn (SMOTE)  
- PyTorch (CPU inference; CUDA optional for experimentation)  
- Matplotlib  

The system is fully functional without GPU acceleration.

---

# 7. Model Architecture

## Superclass Training Model

- Multi-layer Perceptron (Scikit-learn)
- 41-dimensional feature input
- Capped SMOTE for imbalance handling
- Persisted scaler and encoder

Superclass definitions are configurable and can be adjusted without altering the inference pipeline.

---

## Inference Model

- Lightweight feedforward network
- CPU-optimized forward pass
- No GPU dependency required
- Minimal parameter footprint

The model is intentionally small to ensure stable performance on constrained hardware.

---

# 8. Real-World Performance (Full-Day Toll Data)

Evaluation performed on **6,583 samples (entire operational day)**.

## Overall Accuracy

**92.81%**

## Weighted Metrics (Deployment-Representative)

- Weighted Precision: **92.80%**
- Weighted Recall: **92.81%**
- Weighted F1-score: **92.78%**

Because the test set reflects real traffic distribution, weighted metrics closely approximate expected field performance.

Macro F1 (~71%) reflects lower performance on extremely rare vehicle categories — expected in naturally imbalanced toll traffic.

---

# 9. CPU Inference Timing

Average per-sample latency:

- Feature extraction: **10.782 ms**
- Scale + prediction: **0.001 ms**
- End-to-end latency: **10.783 ms**

On CPU-only hardware, the system achieves ~10.8 ms end-to-end latency.

Feature extraction dominates runtime; model inference is effectively negligible.

This confirms suitability for real-time toll plaza deployment.

---

# 10. Benchmark Comparison

| Metric | Image + CNN (CPU) | Feature + MLP (CPU) |
|--------|------------------|---------------------|
| GPU Required | Yes (for acceptable speed) | No |
| CPU Latency | >100 ms | ~10.8 ms |
| Memory Usage | High | Minimal |
| Preprocessing | Image reconstruction | Direct numeric extraction |
| Deployment Fit | Poor | Strong |

The redesigned pipeline reduces latency by an order of magnitude while maintaining high weighted accuracy.

---

# 11. Training Workflow

`python3 train_mlp_model.py`

Steps:

1. Parse `train_super.csv`
2. Extract 41-dimensional features
3. Apply capped SMOTE
4. Train MLP classifier
5. Persist model artifacts

---

# 12. Testing Workflow

`python3 test_mlp_model.py`

Outputs:

- Classification report
- Timestamped prediction CSV
- Timestamped metrics CSV

---

# 13. Engineering Principles

- Architect for actual deployment hardware
- Remove unnecessary abstraction layers
- Model signals in their native domain
- Maintain strict preprocessing parity
- Validate on real traffic distribution

This redesign corrects a hardware–architecture mismatch and aligns modeling with operational constraints.

---

# 14. What This Project Demonstrates

- End-to-end ML system redesign under hardware constraints
- CPU-optimized inference architecture
- Real-world dataset evaluation methodology
- Efficient feature engineering from binary sensor data
- Deployment-driven engineering decisions
- Latency-focused performance validation

