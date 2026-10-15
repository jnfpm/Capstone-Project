import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.preprocessing import RobustScaler

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


def _prepare_features(df):
	features = df.copy()
	passthrough_cols = [
		"Account",
		"Account.1",
		"From Bank",
		"Receiver Bank Name",
		"Receiver Entity ID",
		"Receiver Entity Name",
		"Sender Bank Name",
		"Sender Entity ID",
		"Sender Entity Name",
		"To Bank",
		"Payment Currency",
		"Receiving Currency",
		"Payment Format"
	]
	passthrough_cols = [
		column for column in passthrough_cols if column in features.columns
	]
	features.drop(
		columns=["Type", "Sender Account Matched", "Receiver Account Matched"],
		errors="ignore",
		inplace=True,
	)

	if "Timestamp" in features:
		timestamp = pd.to_datetime(features.pop("Timestamp"))
		features["Timestamp Hour"] = timestamp.dt.hour
		features["Timestamp Day"] = timestamp.dt.day
		features["Timestamp Weekday"] = timestamp.dt.dayofweek

	amount_cols = [
		column
		for column in ("Amount Received", "Amount Paid")
		if column in features.columns
	]
	for column in amount_cols:
		features[column] = np.log1p(features[column].clip(lower=0))

	numeric_cols = [
		column
		for column in features.select_dtypes(include="number").columns
		if column not in passthrough_cols
	]
	return features, numeric_cols, passthrough_cols


def fit_preprocessor(df):
	features, numeric_cols, passthrough_cols = _prepare_features(df)
	transformers = [
		("numeric", RobustScaler(), numeric_cols),
		("passthrough", "passthrough", passthrough_cols),
	]
	preprocessor = ColumnTransformer(
		transformers=transformers,
		remainder="drop",
		verbose_feature_names_out=False,
	)
	# Fit on training data once, then reuse the same transformations on validation data.
	return preprocessor.fit(features)


def preprocess_data(df, fitted_preprocessor):
	features, _, _ = _prepare_features(df)
	transformed = fitted_preprocessor.transform(features)
	if hasattr(transformed, "toarray"):
		transformed = transformed.toarray()
	return pd.DataFrame(
		transformed,
		columns=fitted_preprocessor.get_feature_names_out(),
		index=df.index,
	)



