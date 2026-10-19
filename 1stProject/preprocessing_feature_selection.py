from pathlib import Path

import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from utils import fit_preprocessor, preprocess_data


transactions_df = pd.read_csv(
	Path("data") / "sample_trans.csv",
	low_memory=False,
)

X = transactions_df.drop(columns=["Is Laundering"])
y = transactions_df["Is Laundering"].astype(int)

X_train, X_val, y_train, y_val = train_test_split(
	X,
	y,
	test_size=0.2,
	random_state=42,
	stratify=y,
)

# Fit preprocessing only on training data, then reuse it for validation.
preprocessor = fit_preprocessor(X_train)
X_train = preprocess_data(X_train, preprocessor)
X_val = preprocess_data(X_val, preprocessor)

print(f"Training rows: {len(X_train):,}")
print(f"Validation rows: {len(X_val):,}")
print(f"Training laundering rate: {y_train.mean():.3%}")
print(f"Validation laundering rate: {y_val.mean():.3%}")
print(f"Processed feature count: {X_train.shape[1]:,}")
display(X_train.head())
print("Training columns:")
print(X_train.columns.tolist())