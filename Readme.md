# Implementation Overview

This project implements a fault-injection detection system for Deep Neural Networks using internal layer information and Support Vector Data Description (SVDD).

## Implementation Steps

1. **Prepare Dataset**

   * Load and preprocess the selected dataset.
   * Split the data into training, validation, and test sets.

2. **Train the DNN**

   * Implement or load the selected DNN architecture.
   * Train the model using clean data.
   * Save the trained model and baseline accuracy.

3. **Select Monitoring Layers**

   * Choose several internal DNN layers for monitoring.
   * Register hooks or equivalent mechanisms to capture layer activations during inference.

4. **Extract Internal Features**

   * Run clean samples through the network.
   * Collect activation values from the selected layers.
   * Convert large activation tensors into compact feature vectors using statistical or dimensionality-reduction features.

5. **Create Normal Behaviour Model**

   * Use features extracted from clean executions as normal reference data.
   * Train SVDD models using only normal samples.

6. **Implement Fault Injection**

   * Inject bit flips into model weights and biases.
   * Vary:

     * target layer
     * target parameter
     * bit position
     * number of injected faults
   * Store information about every injected fault.

7. **Run Fault-Injected Inference**

   * Execute the DNN after fault injection.
   * Collect the same internal features from the monitored layers.

8. **Detect Anomalies with SVDD**

   * Compare fault-injected features against the learned normal behaviour.
   * Mark samples outside the SVDD decision boundary as anomalous.

9. **Implement Two Detection Strategies**

   **Multi-layer SVDD**

   * Combine features from several monitored layers.
   * Train one SVDD model using the combined feature vector.

   **Per-layer SVDD**

   * Train a separate SVDD model for each monitored layer.
   * Identify where abnormal behaviour first appears and how it propagates through the network.

10. **Evaluate Detection Performance**

    * Measure:

      * Detection rate
      * False-positive rate
      * Precision
      * Recall
      * F1-score
      * ROC-AUC

11. **Measure System Overhead**

    * Compare normal inference and monitored inference.
    * Measure:

      * Runtime overhead
      * Memory overhead
      * Feature extraction cost
      * SVDD detection cost

12. **Compare the Approaches**

    * Compare multi-layer monitoring against per-layer monitoring.
    * Study which layers provide the most useful information.
    * Identify whether fewer monitored layers can maintain strong detection performance.

13. **Generate Results**

    * Save experiment results in CSV or JSON format.
    * Generate plots, tables, confusion matrices, ROC curves, and layer-wise detection results for thesis analysis.

## Expected Pipeline

```text
Dataset
   ↓
Train DNN
   ↓
Clean Inference
   ↓
Internal Layer Feature Extraction
   ↓
Train SVDD on Normal Behaviour
   ↓
Inject Bit-Flip Faults
   ↓
Fault-Injected Inference
   ↓
Extract Internal Features
   ↓
SVDD Anomaly Detection
   ↓
Multi-Layer / Per-Layer Comparison
   ↓
Performance + Overhead Evaluation
```
