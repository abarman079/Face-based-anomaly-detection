# Face-Based Anomaly Detection Using LFW Dataset

## Project Overview

This project presents a face-based anomaly detection system using the Labeled Faces in the Wild (LFW) dataset. The goal of the project is to identify whether a face image belongs to a known/normal identity group or should be treated as an unknown/anomalous face.

The project compares four machine learning and deep learning approaches:

* KNN Distance Threshold
* Isolation Forest
* MLP Neural Network on FaceNet embeddings
* Convolutional Autoencoder

The final evaluation shows that the MLP model using FaceNet embeddings achieved the strongest overall performance among the four tested models.

## Dataset

The project uses the Labeled Faces in the Wild (LFW) dataset, a public face image dataset commonly used for unconstrained face recognition research.

In this project:

* Normal class: selected high-frequency identities from the LFW dataset
* Anomaly class: other identities treated as unknown or anomalous
* Image input: 96 × 96 RGB face images
* Embedding input: 512-dimensional FaceNet embeddings

## Project Workflow

The project follows this workflow:

1. Load and inspect the LFW face dataset.
2. Analyze identity distribution and class imbalance.
3. Define normal and anomalous identity groups.
4. Preprocess face images.
5. Generate FaceNet embeddings.
6. Train and evaluate four anomaly detection models.
7. Compare model performance using classification and ranking metrics.
8. Visualize results using confusion matrices, ROC curves, precision-recall curves, and model comparison charts.

## Models Used

### 1. KNN Distance Threshold

A distance-based anomaly detection method that compares FaceNet embeddings against known normal samples. A threshold is selected to separate normal and anomalous faces.

### 2. Isolation Forest

An unsupervised anomaly detection method trained on FaceNet embeddings to identify unusual samples based on isolation behavior.

### 3. MLP Neural Network

A supervised neural network classifier trained on FaceNet embeddings. This model achieved the best overall result in the project.

### 4. Convolutional Autoencoder

A reconstruction-based deep learning model trained to reconstruct normal face images. Higher reconstruction error is used as an anomaly signal.

## Final Model Performance

| Model                     | Accuracy | Anomaly Precision | Anomaly Recall | F1-Score | ROC-AUC |
| ------------------------- | -------: | ----------------: | -------------: | -------: | ------: |
| KNN Distance Threshold    |   98.53% |            97.14% |        100.00% |   98.55% |  99.39% |
| Isolation Forest          |   97.94% |            97.11% |         98.82% |   97.96% |  99.23% |
| MLP on FaceNet Embeddings |   98.82% |            98.82% |         98.82% |   98.82% |  99.38% |
| Convolutional Autoencoder |   50.00% |            50.00% |        100.00% |   66.67% |  49.18% |

## Key Finding

The MLP model using FaceNet embeddings produced the best overall performance. It achieved high accuracy, high anomaly precision, high anomaly recall, and the lowest false positive rate among the tested models.

The autoencoder model did not perform well in the final evaluation because it classified all normal samples as anomalies, leading to a very high false positive rate.

## Repository Structure

```text
docs/report/        Final project report in DOCX and PDF format
notebooks/          Main Jupyter Notebook
results/figures/    Generated plots, charts, confusion matrices, and diagrams
results/tables/     CSV result tables and summary tables
models/             Saved trained model files
artifacts/arrays/   NumPy arrays and intermediate model outputs
```

## Important Files

* `notebooks/face-anomaly-detection-using-lfw-4models.ipynb`
  Main notebook containing data processing, model training, evaluation, and visualization.

* `results/tables/24_final_all_four_model_results.csv`
  Final performance comparison of all four models.

* `results/figures/23_final_four_model_performance_comparison.png`
  Final visual comparison of model performance.

* `docs/report/final-report-face-anomaly-detection.pdf`
  Full final report explaining the methodology, results, and conclusion.

## Technologies Used

* Python
* Jupyter Notebook
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* TensorFlow / Keras
* FaceNet embeddings
* LFW dataset

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/abarman079/Face-based-anomaly-detection.git
cd Face-based-anomaly-detection
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open the notebook:

```bash
jupyter notebook notebooks/face-anomaly-detection-using-lfw-4models.ipynb
```

4. Run the notebook cells in order.

## Notes

Large model and array files are tracked using Git LFS. If the files do not download correctly, install Git LFS first:

```bash
git lfs install
git lfs pull
```

## Ethics and Limitations

This project is for academic and research purposes only. It should not be used for real-world identity verification, surveillance, security screening, or decision-making without proper validation, consent, fairness testing, and privacy review.

The model was evaluated on a limited experimental setup using the LFW dataset, so the results may not generalize to real-world face anomaly detection scenarios.

## Author

Abarman079
