# Time-Series Classification — UCI HAR Dataset
**Artificial Neural Networks and Deep Learning** · Politecnico di Milano
 
> Multiclass classification of human physical activities from smartphone sensor time series,
> using a bidirectional LSTM with learned attention pooling.
 
---
 
## Objective
 
Given multivariate time-series data recorded by a smartphone's accelerometer and gyroscope,
the model predicts which of 6 activities a subject is performing:
 
| Label | Activity |
|-------|----------|
| 0 | Walking |
| 1 | Walking Upstairs |
| 2 | Walking Downstairs |
| 3 | Sitting |
| 4 | Standing |
| 5 | Laying |
 
**Evaluation metric:** weighted F1 score on the held-out test set.
 
---
 
## Repository Structure
 
```
.
├── Time_Series_Classifications_Project.ipynb   # Main notebook
├── UCI HAR Dataset/                            # Dataset folder (see below)
│   ├── train/
│   │   ├── X_train.txt
│   │   ├── y_train.txt
│   │   └── subject_train.txt
│   └── test/
│       ├── X_test.txt
│       ├── y_test.txt
│       └── subject_test.txt
├── models/                                     # Saved checkpoints (created at runtime)
└── README.md
```
 
---
 
## Dataset
 
**UCI Human Activity Recognition Using Smartphones**
 
- 30 subjects, 6 activities, 561 features per timestep (accelerometer + gyroscope)
- Pre-segmented windows of 128 timesteps at 50 Hz with 50% overlap
- 7352 training samples, 2947 test samples
**Download:**
```
https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip
```
 
Extract the zip and place the `UCI HAR Dataset/` folder in the same directory as the notebook.
 
Alternatively, on Kaggle the dataset is available directly via *Add Data → UCI HAR Dataset*
with path `/kaggle/input/uci-har-dataset/UCI HAR Dataset`.
 
---
 
## Approach
 
| Step | Detail |
|------|--------|
| Windowing | Sliding window via NumPy stride tricks |
| Augmentation | Gaussian noise, time masking, feature masking (training only) |
| Model | Bidirectional LSTM · LayerNorm · Learned attention · Projection MLP |
| Loss | Focal Loss with class weights and label smoothing |
| Tuning | Grid search with 1-fold fast CV to rank configs |
| Training | 5-fold K-Fold CV on best config, weighted ensemble inference |
| Evaluation | Confusion matrix + per-class F1 on test set |
 
