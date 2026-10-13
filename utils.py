import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LassoCV, LogisticRegression

# FEATURE SELECTION METHODS

# Selection methods return positions and coefficients; names make their results interpretable. Instead of showing like column indices, we can show the actual feature names.
def _get_feature_names(X, feature_names=None):
	if feature_names is None:
		columns = getattr(X, "columns", None)
		if columns is not None:
			feature_names = list(columns)
		else:
			feature_names = [f"feature_{index}" for index in range(X.shape[1])]
	else:
		feature_names = list(feature_names)

	if len(feature_names) != X.shape[1]:
		raise ValueError("feature_names must have one name for each input column")

	return feature_names


def rfe_feature_selection(
	X,
	y,
	feature_names=None,
	n_features_to_select=10,
	step=0.2,
):
	"""Fit RFE with balanced logistic regression and return the fitted selector and ranks."""
	feature_names = _get_feature_names(X, feature_names)
	estimator = LogisticRegression(
		solver="liblinear",
		class_weight="balanced",
		max_iter=2000,
		random_state=42,
	)
	selector = RFE(
		estimator=estimator,
		n_features_to_select=n_features_to_select,
		step=step,
	)
	selector.fit(X, y)

	results = pd.DataFrame({
		"Feature": feature_names,
		"Rank": selector.ranking_,
		"Selected": selector.support_,
	})
	return selector, results


def lasso_regression_feature_selection(
	X,
	y,
	feature_names=None,
	cv=5,
	max_iter=10000,
):
	"""Fit cross-validated Lasso and return the model and its coefficient-based selection."""
	feature_names = _get_feature_names(X, feature_names)
	model = LassoCV(cv=cv, max_iter=max_iter, n_jobs=-1, random_state=42)
	model.fit(X, y)

	results = pd.DataFrame({
		"Feature": feature_names,
		"Coefficient": model.coef_,
		"Selected": model.coef_ != 0,
	})
	return model, results


