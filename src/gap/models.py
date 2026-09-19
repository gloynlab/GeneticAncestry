"""Model training, evaluation, and prediction utilities for GAP ancestry models."""

import ast
import datetime
import os
import time
from itertools import product
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from .utils import *


class GapLearn:
    """Train, evaluate, and apply machine learning models for ancestry prediction."""

    def __init__(
        self,
        task_type: str = "classification",
        multi_class: bool = True,
        score_method: str = "roc_auc",
        feature_standardize: bool = False,
        feature_select: bool = False,
        random_state: int = 42,
    ) -> None:
        """Initialize the learner configuration.

        Args:
            task_type: Either ``"classification"`` or ``"regression"``.
            multi_class: Whether the classification task has more than two
                classes.
            score_method: Scoring metric to use, one of ``"roc_auc"``,
                ``"mse"``, ``"mae"``, or ``"r2"``.
            feature_standardize: Whether to standardize features before
                fitting.
            feature_select: Whether to run feature selection before fitting.
            random_state: Random seed used across models and splits.
        """
        self.time_zone = ZoneInfo("America/Los_Angeles")
        self.time_format = "%Y-%m-%d %H:%M:%S"
        self.score_methods = {
            "roc_auc": roc_auc_score,
            "mse": mean_squared_error,
            "mae": mean_absolute_error,
            "r2": r2_score,
        }
        self.task_type = task_type
        self.multi_class = multi_class
        self.score_method = score_method
        self.feature_standardize = feature_standardize
        self.feature_select = feature_select
        self.random_state = random_state

    def train_cross_val(
        self,
        config_file: str = "config.yaml",
        train_file: str = "features_labeled_train.txt",
        metrics_file: str = "metrics.txt",
        n_features: str = "max",
        target_col: str = "class",
        exclude_columns: List[str] = ["SampleID", "class_name"],
        cv_n_splits: int = 10,
    ) -> None:
        """Run k-fold cross-validation over a grid of models and feature counts.

        For each fold, model, hyperparameter combination, and feature count,
        fits a model and logs its validation score to ``metrics_file``.

        Args:
            config_file: YAML file specifying the models and hyperparameter
                grids to search.
            train_file: Tab-delimited file with training features and labels.
            metrics_file: Output file to append cross-validation metrics to.
            n_features: Comma-separated list of feature counts to try, e.g.
                ``"10,20,30"``; non-numeric entries (e.g. ``"max"``) are kept
                as-is.
            target_col: Name of the label column.
            exclude_columns: Columns to exclude from the feature set besides
                ``target_col``.
            cv_n_splits: Number of cross-validation folds.
        """
        self.config = self.load_yaml(config_file)
        self.df_train = pd.read_table(train_file, header=0, sep="\t")
        self.target_col = target_col
        self.feature_cols = [
            True if x not in [self.target_col] + exclude_columns else False
            for x in self.df_train.columns
        ]
        self.log(f"train shape: {self.df_train.shape}")

        self.cvs = KFold(n_splits=cv_n_splits, shuffle=True, random_state=self.random_state)

        try:
            n_features = [x.strip() for x in n_features.split(",")]
            self.n_features = [int(x) for x in n_features if x.isdigit()] + [
                x for x in n_features if not x.isdigit()
            ]
        except Exception as e:
            print(f"ERROR parsing n_features {e}")

        idx_splits = -1
        for train_idx, val_idx in self.cvs.split(self.df_train):
            idx_splits += 1
            X_train = self.df_train.loc[train_idx, self.feature_cols]
            y_train = self.df_train.loc[train_idx, self.target_col]
            X_val = self.df_train.loc[val_idx, self.feature_cols]
            y_val = self.df_train.loc[val_idx, self.target_col]
            if self.feature_select:
                features = self.feature_selection(X_train, y_train)
            else:
                features = list(X_train.columns)
            X_train_selected = X_train.loc[:, features]
            X_val_selected = X_val.loc[:, features]
            if self.feature_standardize:
                X_train_selected, X_val_selected, scaler = self.feature_standardization(
                    X_train_selected, X_val_selected
                )
            for nfeats in self.n_features:
                if nfeats == "max":
                    nf = X_train_selected.shape[1]
                else:
                    nf = nfeats
                X_train = X_train_selected.iloc[:, 0:nf]
                X_val = X_val_selected.iloc[:, 0:nf]
                for model_name, model_info in self.config["models"].items():
                    model_class = eval(model_info["class"])
                    param_grid = model_info["params"]
                    keys = list(param_grid.keys())
                    combinations = list(product(*param_grid.values()))
                    for combo in combinations:
                        params = dict(zip(keys, combo))
                        model = self.fit_model(
                            model_name, model_class, params, X_train, y_train, X_val, y_val
                        )
                        y_pred = model.predict(X_val)
                        if self.task_type == "classification":
                            y_pred_prob = model.predict_proba(X_val)
                            if self.multi_class:
                                score = self.score_methods[self.score_method](
                                    y_val, y_pred_prob, multi_class="ovr", average="macro"
                                )
                                extra = classification_report(y_val, y_pred, output_dict=True)
                            else:
                                score = self.score_methods[self.score_method](
                                    y_val, y_pred_prob[:, 1]
                                )
                                extra = confusion_matrix(y_val, y_pred).ravel()
                                extra = ",".join([str(x) for x in extra])
                        elif self.task_type == "regression":
                            score = self.score_methods[self.score_method](y_val, y_pred)
                            extra = "."
                        self.log_metrics(
                            [model_name, str(nfeats), str(params), idx_splits]
                            + [self.score_method, score, "val", extra],
                            out_file=metrics_file,
                        )

    def feature_selection(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        remove_low_variance: bool = True,
        remove_high_correlation: bool = True,
        sort_by_mutual_info_score: bool = True,
    ) -> List[str]:
        """Select and rank features for model training.

        Args:
            X_train: Training feature matrix.
            y_train: Training labels, used for mutual-information ranking.
            remove_low_variance: Whether to drop near-constant features.
            remove_high_correlation: Whether to drop highly correlated
                features.
            sort_by_mutual_info_score: Whether to sort remaining features by
                mutual information with ``y_train``.

        Returns:
            The names of the selected features, in the chosen order.
        """
        if remove_low_variance:
            X_train = self._remove_low_variance(X_train)
        if remove_high_correlation:
            X_train = self._remove_high_correlation(X_train)
        if sort_by_mutual_info_score:
            features = self._sort_by_mutual_info_score(X_train, y_train)
        else:
            features = list(X_train.columns)
        return features

    def feature_standardization(
        self,
        X: pd.DataFrame,
        X2: Optional[pd.DataFrame] = None,
        scaler: Optional[StandardScaler] = None,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], StandardScaler]:
        """Standardize features using a (possibly pre-fit) scaler.

        Args:
            X: Feature matrix to fit (if ``scaler`` is ``None``) and transform.
            X2: Optional second feature matrix (e.g. validation/test set) to
                transform using the same scaler.
            scaler: An existing fitted :class:`~sklearn.preprocessing.StandardScaler`.
                If ``None``, a new scaler is fit on ``X``.

        Returns:
            A tuple of ``(X_scaled, X2_scaled, scaler)``, where ``X2_scaled``
            is ``None`` if ``X2`` was not provided.
        """
        if not scaler:
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
            if X2 is not None:
                X2 = pd.DataFrame(scaler.transform(X2), columns=X2.columns)
        else:
            X = pd.DataFrame(scaler.transform(X), columns=X.columns)
            if X2 is not None:
                X2 = pd.DataFrame(scaler.transform(X2), columns=X2.columns)
        return X, X2, scaler

    def fit_model(
        self,
        model_name: str,
        model_class: type,
        params: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> Any:
        """Instantiate and fit a model, using validation data for early stopping where supported.

        Args:
            model_name: Key identifying the model type (e.g. ``"xgb"``,
                ``"catboost"``, ``"lgb"``, ``"mlp"``, ``"random_forest"``,
                ``"svm"``), used to select model-specific fit arguments.
            model_class: The model's class, to be instantiated with ``params``.
            params: Hyperparameters passed to ``model_class``.
            X_train: Training feature matrix.
            y_train: Training labels.
            X_val: Optional validation feature matrix, used for early stopping
                by boosting models.
            y_val: Optional validation labels, paired with ``X_val``.

        Returns:
            The fitted model instance.
        """
        if model_name == "mlp":
            model = model_class(**params, random_state=self.random_state, verbose=0)
            model.fit(X_train, y_train)
        elif model_name == "catboost":
            model = model_class(**params, random_state=self.random_state, verbose=0)
            if y_val is not None:
                model.fit(X_train, y_train, eval_set=(X_val, y_val), logging_level="Silent")
            else:
                model.fit(X_train, y_train, logging_level="Silent")
        elif model_name == "lgb":
            model = model_class(**params, random_state=self.random_state, verbose=-1)
            if y_val is not None:
                model.fit(X_train, y_train, eval_set=(X_val, y_val))
            else:
                model.fit(X_train, y_train)
        elif model_name == "xgb":
            model = model_class(**params, random_state=self.random_state)
            if y_val is not None:
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            else:
                model.fit(X_train, y_train, verbose=False)
        elif model_name == "random_forest":
            model = model_class(**params, random_state=self.random_state, verbose=0)
            model.fit(X_train, y_train)
        elif model_name == "svm":
            model = model_class(
                **params, probability=True, random_state=self.random_state, verbose=0
            )
            model.fit(X_train, y_train)
        else:
            model = model_class(**params, random_state=self.random_state)
            model.fit(X_train, y_train)
        return model

    def get_best_model_and_params(
        self, metrics_file: str = "metrics.txt", dataset: str = "val"
    ) -> None:
        """Aggregate cross-validation metrics and select the best hyperparameters per model.

        Reads ``metrics_file``, averages scores across CV folds for each
        combination of feature count, model, and hyperparameters, writes the
        full ranking to ``<metrics_file>_sorted.txt``, and writes the top
        entry per model (at the best feature count) to
        ``<metrics_file>_sorted_best.txt``.

        Args:
            metrics_file: Path to the raw cross-validation metrics file
                produced by :meth:`train_cross_val`.
            dataset: Which logged dataset split to aggregate (typically
                ``"val"``).
        """
        self.metrics = pd.read_table(metrics_file, header=0, sep="\t")
        if self.metrics.shape[0]:
            L = []
            df1 = self.metrics.loc[self.metrics["dataset"] == dataset,]
            for n_features in df1["n_features"].unique():
                df2 = df1.loc[df1["n_features"] == n_features,]
                for model in df2["model"].unique():
                    df3 = df2.loc[df2["model"] == model,]
                    for params in df3["params"].unique():
                        df4 = df3.loc[df3["params"] == params,]
                        if self.task_type == "classification":
                            if self.multi_class:
                                extra = "."
                            else:
                                extra = (
                                    df4["extra"]
                                    .str.split(",", expand=True)
                                    .astype(int)
                                    .sum(axis=0)
                                )
                                extra = ",".join([str(x) for x in extra])
                        elif self.task_type == "regression":
                            extra = "."
                        sm = np.mean(df4["score"])
                        L.append(
                            [n_features, model, params] + [self.score_method, sm, dataset, extra]
                        )

            out_file = f'{metrics_file.replace(".txt", "_sorted.txt")}'
            out_file2 = f'{metrics_file.replace(".txt", "_sorted_best.txt")}'
            df = pd.DataFrame(L)
            df.columns = ["n_features", "model", "params"] + [
                "score_method",
                "score",
                "dataset",
                "extra",
            ]
            if self.score_method in ["roc_auc", "r2"]:
                df.sort_values(by="score", ascending=False, inplace=True)
            elif self.score_method in ["mse", "mae"]:
                df.sort_values(by="score", ascending=True, inplace=True)
            df.to_csv(out_file, header=True, index=False, sep="\t")

            df2 = df.drop_duplicates(subset=["model"], keep="first")
            df2.to_csv(out_file2, header=True, index=False, sep="\t")
            self.log("best model params:")
            print(df3)

    def final_fit_eval_on_full_train_then_eval_on_test(
        self,
        config_file: str = "config.yaml",
        metrics_file: str = "metrics_sorted_best.txt",
        train_file: str = "features_labeled_train.txt",
        test_file: str = "features_labeled_test.txt",
        exclude_columns: List[str] = ["SampleID", "class_name"],
        target_col: str = "class",
        ensemble: bool = False,
        ensemble_voting: str = "soft",
        ensemble_calibration: bool = False,
        calibration_method: str = "sigmoid",
        calibration_cv: int = 5,
        model_file: str = "model.pkl",
    ) -> None:
        """Refit selected models on the full training set and evaluate on the test set.

        For each model listed in ``metrics_file``, refits it on the full
        training set (optionally standardizing/selecting features), evaluates
        it on both train and test sets, and saves all fitted models (plus
        optional ensemble) to ``model_file``.

        Args:
            config_file: YAML file specifying model classes.
            metrics_file: File (typically ``*_sorted_best.txt``) listing the
                model names, feature counts, and hyperparameters to refit.
            train_file: Tab-delimited file with training features and labels.
            test_file: Tab-delimited file with test features and labels.
            exclude_columns: Columns to exclude from the feature set besides
                ``target_col``.
            target_col: Name of the label column.
            ensemble: Whether to also fit a voting ensemble of all models.
            ensemble_voting: Voting strategy for the ensemble (``"soft"`` or
                ``"hard"``).
            ensemble_calibration: Whether to calibrate each model before
                ensembling.
            calibration_method: Calibration method, passed to
                :class:`~sklearn.calibration.CalibratedClassifierCV`.
            calibration_cv: Number of CV folds used for calibration.
            model_file: Output path to save the fitted models (via
                :mod:`joblib`).
        """
        self.config = self.load_yaml(config_file)
        df = pd.read_table(metrics_file, header=0, sep="\t")
        self.df_train = pd.read_table(train_file, header=0, sep="\t")
        self.df_test = pd.read_table(test_file, header=0, sep="\t")
        self.target_col = target_col
        self.feature_cols = [
            True if x not in [self.target_col] + exclude_columns else False
            for x in self.df_train.columns
        ]

        to_save = {}
        X_train_full = self.df_train.loc[:, self.feature_cols]
        y_train = self.df_train.loc[:, self.target_col]
        X_test_full = self.df_test.loc[:, self.feature_cols]
        y_test = self.df_test.loc[:, self.target_col]
        if self.feature_select:
            features_full = self.feature_selection(X_train_full, y_train)
        else:
            features_full = list(X_train_full.columns)

        models = {}
        out_file = metrics_file.replace(".txt", "_eval.txt")
        self.log("final eval on full train and test:")

        for n in range(df.shape[0]):
            if ensemble:
                n_features = df["n_features"].iloc[0]
            else:
                n_features = df["n_features"].iloc[n]

            if n_features == "max":
                n_features = len(features_full)
            else:
                n_features = int(n_features)

            features = features_full[0:n_features]
            X_train = X_train_full.loc[:, features]
            X_test = X_test_full.loc[:, features]

            if self.feature_standardize:
                X_train, X_test, scaler = self.feature_standardization(X_train, X_test)
            else:
                scaler = None

            params = ast.literal_eval(df["params"].iloc[n])
            model_name = df["model"].iloc[n]
            model_class = eval(self.config["models"][model_name]["class"])

            model = self.fit_model(model_name, model_class, params, X_train, y_train)
            models[model_name] = model
            self.eval_model(model, X_train, y_train, model_name, n_features, "train", out_file)
            self.eval_model(model, X_test, y_test, model_name, n_features, "test", out_file)

            to_save[model_name] = model
            to_save[f"{model_name}_n_features"] = n_features
            to_save[f"{model_name}_features"] = features
            to_save[f"{model_name}_scaler"] = scaler

        if ensemble:
            model_name = "ensemble"
            if ensemble_calibration:
                if self.task_type == "classification":
                    for k in models:
                        models[k] = CalibratedClassifierCV(
                            models[k], method=calibration_method, cv=calibration_cv
                        )
                else:
                    self.log("ensemble calibration is only supported for classification tasks")

            if self.task_type == "classification":
                model = VotingClassifier(
                    estimators=list(models.items()), voting=ensemble_voting
                ).fit(X_train, y_train)
            elif self.task_type == "regression":
                model = VotingRegressor(
                    estimators=list(models.items()), voting=ensemble_voting
                ).fit(X_train, y_train)

            self.eval_model(model, X_train, y_train, model_name, n_features, "train", out_file)
            self.eval_model(model, X_test, y_test, model_name, n_features, "test", out_file)
            to_save[model_name] = model
            to_save[f"{model_name}_n_features"] = n_features
            to_save[f"{model_name}_features"] = features
            to_save[f"{model_name}_scaler"] = scaler

        joblib.dump(to_save, model_file)

    def predict(
        self,
        in_file: str = "to_predict_features.txt",
        out_file: str = "predicted.txt",
        label_file: str = "features_labels.txt",
        model_name: str = "random_forrest",
        model_file: str = "model.pkl",
        include_columns: List[str] = ["SampleID"],
    ) -> None:
        """Predict ancestry classes for new samples using a saved model.

        Args:
            in_file: Tab-delimited file with the feature matrix to predict on.
            out_file: Output file to write predictions and class probabilities
                to.
            label_file: File mapping numeric class codes to class names.
            model_name: Key of the model to use, as saved in ``model_file``.
            model_file: Path to the ``joblib``-serialized dictionary of
                fitted models, features, and scalers.
            include_columns: Columns from ``in_file`` to carry through into
                the output.
        """
        try:
            df_labels = pd.read_table(label_file, header=0, sep="\t")
            num2name = dict(zip(df_labels["class"], df_labels["class_name"]))
        except Exception as e:
            print(f"ERROR loading label file {e}")
            num2name = {}

        M = joblib.load(model_file)
        features = list(M[f"{model_name}_features"])
        scaler = M[f"{model_name}_scaler"]
        model = M[model_name]

        df = pd.read_table(in_file, header=0, sep="\t")
        X = df.loc[:, features]

        if self.feature_standardize:
            X, _, _ = self.feature_standardization(X, scaler=scaler)

        if list(X.columns) != list(features):
            self.log("ERROR features are different")

        y_pred = model.predict(X)
        y_pred_prob = model.predict_proba(X)

        df_y_pred = pd.DataFrame(y_pred)
        df_y_pred.columns = ["class"]
        df_y_pred_prob = pd.DataFrame(y_pred_prob)

        if include_columns:
            df_res = pd.concat([df[include_columns], df_y_pred, df_y_pred_prob], axis=1)
        else:
            df_res = pd.concat([df_y_pred, df_y_pred_prob], axis=1)

        try:
            df_res["class"] = df_res["class"].map(num2name)
            df_res.columns = ["SampleID", "class"] + [
                num2name.get(x, ".") for x in df_res.columns[2:]
            ]
        except Exception as e:
            print(f"ERROR mapping class names {e}")

        df_res.to_csv(out_file, header=True, index=False, sep="\t")

        self.log("final predicted results:")
        print(df_res.tail(10))

    def eval_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        n_features: int,
        dataset: str,
        out_file: str = "eval.txt",
    ) -> None:
        """Score a fitted model on a dataset and append the result to a metrics file.

        Args:
            model: A fitted model exposing ``predict`` (and ``predict_proba``
                for classification).
            X: Feature matrix to evaluate on.
            y: True labels/targets for ``X``.
            model_name: Name of the model, for logging.
            n_features: Number of features used, for logging.
            dataset: Name of the dataset split being evaluated (e.g.
                ``"train"``, ``"test"``).
            out_file: File to append the evaluation metrics to.
        """
        y_pred = model.predict(X)
        if self.task_type == "classification":
            y_pred_prob = model.predict_proba(X)
            if self.multi_class:
                score = self.score_methods[self.score_method](
                    y, y_pred_prob, multi_class="ovr", average="macro"
                )
                extra = classification_report(y, y_pred, output_dict=True)
            else:
                score = self.score_methods[self.score_method](y, y_pred_prob[:, 1])
                extra = confusion_matrix(y, y_pred).ravel()
                extra = ",".join([str(x) for x in extra])
        elif self.task_type == "regression":
            score = self.score_methods[self.score_method](y, y_pred)
            extra = "."
        self.log_metrics(
            [model_name, n_features, ".", ".", self.score_method, score, dataset, extra],
            out_file=out_file,
        )

    def _remove_low_variance(self, X: pd.DataFrame, threshold: float = 1e-3) -> pd.DataFrame:
        """Drop features whose variance is below a threshold.

        Args:
            X: Feature matrix.
            threshold: Minimum variance required to keep a feature.

        Returns:
            The feature matrix with low-variance columns removed.
        """
        selector = VarianceThreshold(threshold=threshold)
        X = pd.DataFrame(selector.fit_transform(X), columns=X.columns[selector.get_support()])
        return X

    def _remove_high_correlation(self, X: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """Drop features that are highly correlated with another feature.

        Args:
            X: Feature matrix.
            threshold: Absolute correlation above which one of a pair of
                features is dropped.

        Returns:
            The feature matrix with redundant columns removed.
        """
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] >= threshold)]
        X.drop(columns=to_drop, inplace=True)
        return X

    def _sort_by_mutual_info_score(
        self, X: pd.DataFrame, y: pd.Series, discrete_features: str = "auto"
    ) -> List[str]:
        """Rank features by mutual information with the target.

        Args:
            X: Feature matrix.
            y: Target labels.
            discrete_features: Passed to
                :func:`~sklearn.feature_selection.mutual_info_classif` to
                indicate which features are discrete.

        Returns:
            Feature names sorted by descending mutual information score.
        """
        mi = mutual_info_classif(
            X, y, discrete_features=discrete_features, random_state=self.random_state
        )
        df = pd.DataFrame()
        df["features"] = X.columns
        df["mi_score"] = mi
        df.sort_values(by="mi_score", ascending=False, inplace=True)
        return list(df["features"])

    def log_metrics(self, L: List[Any], out_file: str = "metrics.txt") -> None:
        """Append a timestamped metrics row to a tab-delimited file, writing a header if new.

        Args:
            L: Row values to log, in the order
                ``[model, n_features, params, idx_splits, score_method,
                score, dataset, extra]``.
            out_file: Path of the metrics file to write/append to.
        """
        current_time = (
            datetime.datetime.now().astimezone(self.time_zone).strftime(self.time_format)
        )
        line = "\t".join([current_time] + [str(x) for x in L])
        self.log(line)

        header = [
            "time",
            "model",
            "n_features",
            "params",
            "idx_splits",
            "score_method",
            "score",
            "dataset",
            "extra",
        ]

        if out_file not in vars(self):
            setattr(self, out_file, True)
            with open(out_file, "w") as f:
                f.write("\t".join(header) + "\n")
                f.write(line + "\n")
        else:
            with open(out_file, "a") as f:
                f.write(line + "\n")

    def log(self, txt: str = "") -> None:
        """Print a message prefixed with the current timestamp.

        Args:
            txt: Message to log.
        """
        current_time = (
            datetime.datetime.now().astimezone(self.time_zone).strftime(self.time_format)
        )
        print(f"{current_time} {txt}", flush=True)

    def load_yaml(self, config_file: str) -> Dict[str, Any]:
        """Load a YAML configuration file, falling back to the packaged config directory.

        Args:
            config_file: Path to the YAML config file. If it does not exist
                relative to the current directory, it is looked up under the
                package's ``config`` directory instead.

        Returns:
            The parsed YAML content, or an empty dict if loading failed.
        """
        config = {}
        if not os.path.exists(config_file):
            config_file = config_dir + "/" + config_file
        config_file = os.path.expanduser(config_file)
        self.log(f"using config {config_file}")

        try:
            with open(config_file) as f:
                config = yaml.safe_load(f)
        except Exception as e:
            self.log(f"ERROR loading config {e}")
        return config


if __name__ == "__main__":
    gl = GapLearn()
    gl.train_cross_val()
    gl.get_best_model_and_params()
    gl.final_fit_eval_on_full_train_then_eval_on_test()
    gl.predict()
