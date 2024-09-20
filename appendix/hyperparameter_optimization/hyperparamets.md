Model Optimize Range of values
Linear Regression

- fit_intercept
- normalize
- True/False
- True/False
  Ridge
- alpha
- fit_intercept
- normalize
- 0.01, 0.1, 1.0, 10, 100
- True/False
- True/False
  k-neighbors
- n_neighbors
- p
- 2, 4, 8, 16 ….
- 2, 3
  SVM
- C
- gamma
- class_weight
- 0.001,0.01..10..100..1000
- ‘auto’, RS\*
- ‘balanced’ , None
  Logistic Regression
- Penalty
- C
- l1 or l2
- 0.001, 0.01…..10...100
  Lasso
- Alpha
- Normalize
- 0.1, 1.0, 10
- True/False
  Random Forest
- n_estimators
- max_depth
- min_samples_split
- min_samples_leaf
- max features
- 120, 300, 500, 800, 1200
- 5, 8, 15, 25, 30, None
- 1, 2, 5, 10, 15, 100
- 1, 2, 5, 10
- log2, sqrt, None
  XGBoost
- eta
- gamma
- max_depth
- min_child_weight
- subsample
- colsample_bytree
- lambda
- alpha
- 0.01,0.015, 0.025, 0.05, 0.1
- 0.05-0.1,0.3,0.5,0.7,0.9,1.0
- 3, 5, 7, 9, 12, 15, 17, 25
- 1, 3, 5, 7
- 0.6, 0.7, 0.8, 0.9, 1.0
- 0.6, 0.7, 0.8, 0.9, 1.0
- 0.01-0.1, 1.0 , RS\*
- 0, 0.1, 0.5, 1.0 RS\*
