# MSCS 634 – Lab 2: Classification Using KNN and RNN

**Name:** Sandeep Reddy Banda  
**Course:** Advanced Big Data and Data Mining (MSCS 634)  
**University:** University of the Cumberlands  
**Instructor:** Satish Penmatsa

## Purpose
In this lab I compared K-Nearest Neighbors (KNN) and Radius Neighbors (RNN) classifiers on the Wine dataset from scikit-learn (178 samples, 13 features, 3 classes). I used an 80/20 train/test split and tested different `k` values (1, 5, 11, 15, 21) for KNN and different radius values (350, 400, 450, 500, 550, 600) for RNN to see how the parameter choice affects accuracy.

## Files
- `Lab2_KNN_RNN.ipynb` – the full notebook with code, outputs, plots and written analysis
- `README.md` – this file

## Results

| k | KNN accuracy |
|---|---|
| 1 | 0.778 |
| 5 | 0.722 |
| 11 | 0.750 |
| 15 | 0.750 |
| 21 | 0.778 |

| Radius | RNN accuracy |
|---|---|
| 350 | 0.750 |
| 400 | 0.722 |
| 450 | 0.722 |
| 500 | 0.722 |
| 550 | 0.722 |
| 600 | 0.722 |

## Key Insights
- KNN accuracy stayed between 72% and 78% with no clear trend as k changed. With only 36 test samples, one prediction is about 2.8%, so the differences between k values are small.
- RNN accuracy dropped from 75% to 72.2% and then stayed flat. Even at radius 350, each test point already had about 78 training points inside the radius, so a bigger radius did not change the votes much.
- Both models scored about the same, with KNN only one prediction ahead at its best setting.
- The features are on very different scales (`proline` goes up to about 1680, other features stay below 3), so `proline` dominates the distance. As an extra check, I standardized the features and KNN accuracy jumped to about 94–97%. Scaling mattered more than choosing k or the radius.
- KNN is a safer default because it always uses exactly k neighbors. RNN can work well when point density is uneven, but the radius has to match the scale of the data.

## Challenges and Decisions
- With a fixed radius, a test point can have no neighbors, and `RadiusNeighborsClassifier` throws an error. I set `outlier_label='most_frequent'` to handle that. In my runs no test points actually ended up with zero neighbors, but I kept it to be safe.
- I used `random_state=42` for the split so the results can be reproduced. Since it is only one split, cross-validation would give a more reliable comparison.
- The extra scaling test was not required, but I added it to check my guess about why accuracy was low.
