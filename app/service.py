from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from xgboost import XGBClassifier

from .schemas import (
    DSO1PredictRequest,
    DSO2PredictRequest,
    DSO3PredictRequest,
    TransactionBase,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "accounting_data.csv"
ARTIFACT_DIR = PROJECT_ROOT / "backend" / "artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "dso_model_artifacts.joblib"
ARTIFACT_VERSION = 1

ACCOUNT_TYPE_MAPPING = {"Asset": 0, "Liability": 1, "Revenue": 2, "Expense": 3}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_from_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(records)
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["Date"] = frame["Date"].map(pd.Timestamp.toordinal)
    frame["Account Type"] = (
        frame["Account Type"]
        .map(ACCOUNT_TYPE_MAPPING)
        .fillna(-1)
        .astype(int)
    )
    frame["Missing Data Indicator"] = frame[
        "Missing Data Indicator"
    ].astype(int)
    return frame


def _frame_from_source(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["Date"] = frame["Date"].map(pd.Timestamp.toordinal)
    frame["Account Type"] = (
        frame["Account Type"]
        .map(ACCOUNT_TYPE_MAPPING)
        .fillna(-1)
        .astype(int)
    )
    frame["Missing Data Indicator"] = frame[
        "Missing Data Indicator"
    ].astype(int)
    return frame


@dataclass(slots=True)
class DSO1Bundle:
    model: Any
    feature_columns: list[str]
    X_test: pd.DataFrame
    y_test: pd.Series
    y_pred: np.ndarray
    y_proba: np.ndarray


@dataclass(slots=True)
class DSO2Bundle:
    model: Any
    feature_columns: list[str]
    X_test: pd.DataFrame
    y_test: pd.Series
    y_pred: np.ndarray


@dataclass(slots=True)
class DSO3Bundle:
    scaler: StandardScaler
    pca: PCA
    pca_eigenvalues: np.ndarray
    pca_explained_variance_ratio: np.ndarray
    pca_kaiser_components: int
    kmeans: KMeans
    dbscan: DBSCAN
    nn: NearestNeighbors
    feature_columns: list[str]
    X_test: pd.DataFrame
    projection_test: np.ndarray
    y_clusters_test: np.ndarray
    y_anomaly_test: np.ndarray
    eps: float
    silhouette: float
    train_projection: np.ndarray
    train_clusters: np.ndarray


class ModelService:
    def __init__(self, data_path: Path = DATA_PATH) -> None:
        self.data_path = data_path
        self.raw_df = pd.read_csv(self.data_path)
        self.prepared_df = _frame_from_source(self.raw_df)
        self.data_signature = _file_sha256(self.data_path)

        persisted = self._load_persisted_state()
        if persisted is None:
            self.dso1 = self._build_dso1()
            self.dso2 = self._build_dso2()
            self.dso3 = self._build_dso3()
            self._save_persisted_state()
        else:
            self.dso1 = persisted["dso1"]
            self.dso2 = persisted["dso2"]
            self.dso3 = persisted["dso3"]

    def _artifact_metadata(self) -> dict[str, Any]:
        return {
            "version": ARTIFACT_VERSION,
            "data_signature": self.data_signature,
            "data_path": str(self.data_path),
        }

    def _load_persisted_state(self) -> dict[str, Any] | None:
        if not ARTIFACT_PATH.exists():
            return None

        try:
            payload = joblib.load(ARTIFACT_PATH)
        except Exception:
            return None

        if not isinstance(payload, dict):
            return None

        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            return None

        if metadata.get("version") != ARTIFACT_VERSION:
            return None

        if metadata.get("data_signature") != self.data_signature:
            return None

        dso1 = payload.get("dso1")
        dso2 = payload.get("dso2")
        dso3 = payload.get("dso3")
        if (
            not isinstance(dso1, DSO1Bundle)
            or not isinstance(dso2, DSO2Bundle)
            or not isinstance(dso3, DSO3Bundle)
        ):
            return None

        return {"dso1": dso1, "dso2": dso2, "dso3": dso3}

    def _save_persisted_state(self) -> None:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": self._artifact_metadata(),
            "dso1": self.dso1,
            "dso2": self.dso2,
            "dso3": self.dso3,
        }
        joblib.dump(payload, ARTIFACT_PATH)

    def _dso1_feature_frame(self) -> pd.DataFrame:
        columns = [
            "Date",
            "Account Type",
            "Transaction Amount",
            "Cash Flow",
            "Net Income",
            "Revenue",
            "Expenditure",
            "Profit Margin",
            "Debt-to-Equity Ratio",
            "Operating Expenses",
            "Gross Profit",
            "Transaction Volume",
            "Processing Time (seconds)",
            "Accuracy Score",
            "Missing Data Indicator",
            "Normalized Transaction Amount",
        ]
        return self.prepared_df[columns].copy()

    def _dso2_feature_frame(self) -> pd.DataFrame:
        columns = [
            "Date",
            "Account Type",
            "Transaction Amount",
            "Cash Flow",
            "Net Income",
            "Expenditure",
            "Profit Margin",
            "Debt-to-Equity Ratio",
            "Operating Expenses",
            "Gross Profit",
            "Transaction Volume",
            "Processing Time (seconds)",
            "Accuracy Score",
            "Missing Data Indicator",
            "Normalized Transaction Amount",
        ]
        return self.prepared_df[columns].copy()

    def _dso3_feature_frame(self) -> pd.DataFrame:
        columns = [
            "Date",
            "Account Type",
            "Transaction Amount",
            "Cash Flow",
            "Net Income",
            "Revenue",
            "Expenditure",
            "Profit Margin",
            "Debt-to-Equity Ratio",
            "Operating Expenses",
            "Gross Profit",
            "Transaction Volume",
            "Processing Time (seconds)",
            "Accuracy Score",
            "Missing Data Indicator",
            "Normalized Transaction Amount",
        ]
        return self.prepared_df[columns].copy()

    def _build_dso1(self) -> DSO1Bundle:
        X = self._dso1_feature_frame()
        y = self.prepared_df["Transaction Outcome"].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model = XGBClassifier(
            objective="binary:logistic",
            max_depth=4,
            learning_rate=0.1,
            n_estimators=120,
            eval_metric="logloss",
            random_state=42,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        return DSO1Bundle(
            model=model,
            feature_columns=list(X.columns),
            X_test=X_test.reset_index(drop=True),
            y_test=y_test.reset_index(drop=True),
            y_pred=y_pred,
            y_proba=y_proba,
        )

    def _build_dso2(self) -> DSO2Bundle:
        X = self._dso2_feature_frame()
        y = self.prepared_df["Revenue"].astype(float)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0
        )
        model = Pipeline(
            steps=[
                (
                    "poly",
                    PolynomialFeatures(degree=2, include_bias=False),
                ),
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression()),
            ]
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return DSO2Bundle(
            model=model,
            feature_columns=list(X.columns),
            X_test=X_test.reset_index(drop=True),
            y_test=y_test.reset_index(drop=True),
            y_pred=y_pred,
        )

    def _infer_dso3_anomaly(
        self,
        projection_row: np.ndarray,
        eps: float,
        dbscan_labels: np.ndarray,
        nn: NearestNeighbors,
    ) -> bool:
        nearest_distance, nearest_index = nn.kneighbors(
            np.asarray(projection_row).reshape(1, -1)
        )
        nearest_distance_value = float(nearest_distance[0, 0])
        nearest_label = int(dbscan_labels[nearest_index[0, 0]])
        return bool(
            nearest_distance_value > eps or nearest_label == -1
        )

    def _build_dso3(self) -> DSO3Bundle:
        X = self._dso3_feature_frame()
        y = self.prepared_df["Transaction Outcome"].astype(int)
        X_train, X_test, _, _ = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        pca_full = PCA(random_state=42)
        pca_full.fit(X_train_scaled)
        pca = PCA(n_components=2, random_state=42)
        train_projection = pca.fit_transform(X_train_scaled)
        test_projection = pca.transform(X_test_scaled)
        pca_eigenvalues = pca_full.explained_variance_
        pca_explained_variance_ratio = (
            pca_full.explained_variance_ratio_
        )
        pca_kaiser_components = int(
            np.sum(pca_eigenvalues >= 1.0)
        )

        best_score = float("-inf")
        best_model: KMeans | None = None
        best_labels: np.ndarray | None = None
        for k in range(2, 9):
            candidate = KMeans(n_clusters=k, n_init=12, random_state=42)
            labels = candidate.fit_predict(train_projection)
            if len(np.unique(labels)) > 1:
                score = float(
                    silhouette_score(train_projection, labels)
                )
            else:
                score = float("-inf")
            if score > best_score:
                best_score = score
                best_model = candidate
                best_labels = labels

        assert best_model is not None and best_labels is not None

        neighbor_probe = NearestNeighbors(n_neighbors=5)
        neighbor_probe.fit(train_projection)
        distances, _ = neighbor_probe.kneighbors(train_projection)
        eps = float(np.percentile(np.sort(distances[:, -1]), 90))

        dbscan = DBSCAN(eps=eps, min_samples=5)
        dbscan.fit(train_projection)

        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(train_projection)

        y_clusters_test = best_model.predict(test_projection)
        y_anomaly_test = np.array(
            [
                self._infer_dso3_anomaly(
                    point,
                    eps=eps,
                    dbscan_labels=dbscan.labels_,
                    nn=nn,
                )
                for point in test_projection
            ],
            dtype=bool,
        )

        return DSO3Bundle(
            scaler=scaler,
            pca=pca,
            pca_eigenvalues=pca_eigenvalues,
            pca_explained_variance_ratio=pca_explained_variance_ratio,
            pca_kaiser_components=pca_kaiser_components,
            kmeans=best_model,
            dbscan=dbscan,
            nn=nn,
            feature_columns=list(X.columns),
            X_test=X_test.reset_index(drop=True),
            projection_test=test_projection,
            y_clusters_test=y_clusters_test,
            y_anomaly_test=y_anomaly_test,
            eps=eps,
            silhouette=best_score,
            train_projection=train_projection,
            train_clusters=best_labels,
        )

    def _to_frame(
        self,
        payload: (
            TransactionBase
            | DSO1PredictRequest
            | DSO2PredictRequest
            | DSO3PredictRequest
        ),
    ) -> pd.DataFrame:
        return _frame_from_records([payload.model_dump(by_alias=True)])

    def _dso1_features(self, row: pd.DataFrame) -> pd.DataFrame:
        return row[self.dso1.feature_columns]

    def _dso2_features(self, row: pd.DataFrame) -> pd.DataFrame:
        return row[self.dso2.feature_columns]

    def _dso3_features(self, row: pd.DataFrame) -> pd.DataFrame:
        return row[self.dso3.feature_columns]

    def predict_dso1(
        self, payload: DSO1PredictRequest
    ) -> dict[str, Any]:
        row = self._to_frame(payload)
        features = self._dso1_features(row)
        probabilities = (
            self.dso1.model.predict_proba(features)[0]
        )
        label = int(self.dso1.model.predict(features)[0])
        return {
            "model": "XGBoostClassifier",
            "predicted_label": label,
            "predicted_class": (
                "Succès" if label == 1 else "Échec"
            ),
            "probability_success": float(probabilities[1]),
            "probability_failure": float(probabilities[0]),
            "confidence": float(np.max(probabilities)),
        }

    def predict_dso2(
        self, payload: DSO2PredictRequest
    ) -> dict[str, Any]:
        row = self._to_frame(payload)
        features = self._dso2_features(row)
        predicted_value = float(
            self.dso2.model.predict(features)[0]
        )
        return {
            "model": "PolynomialRegression(degree=2)",
            "predicted_value": predicted_value,
            "feature_count": len(self.dso2.feature_columns),
        }

    def predict_dso3(
        self, payload: DSO3PredictRequest
    ) -> dict[str, Any]:
        row = self._to_frame(payload)
        features = self._dso3_features(row)
        scaled = self.dso3.scaler.transform(features)
        projection = self.dso3.pca.transform(scaled)
        cluster_id = int(
            self.dso3.kmeans.predict(projection)[0]
        )
        nearest_distance, nearest_index = (
            self.dso3.nn.kneighbors(projection)
        )
        distance = float(nearest_distance[0, 0])
        nearest_label = int(
            self.dso3.dbscan.labels_[nearest_index[0, 0]]
        )
        anomaly = bool(
            distance > self.dso3.eps or nearest_label == -1
        )
        return {
            "model": "PCA + KMeans + DBSCAN",
            "cluster_id": cluster_id,
            "is_anomaly": anomaly,
            "anomaly_score": float(
                distance / max(self.dso3.eps, 1e-9)
            ),
            "pca_coordinates": [
                float(projection[0, 0]),
                float(projection[0, 1]),
            ],
            "nearest_distance": distance,
        }

    def batch_predict_dso1(
        self, rows: list[DSO1PredictRequest]
    ) -> list[dict[str, Any]]:
        return [self.predict_dso1(row) for row in rows]

    def batch_predict_dso2(
        self, rows: list[DSO2PredictRequest]
    ) -> list[dict[str, Any]]:
        return [self.predict_dso2(row) for row in rows]

    def batch_predict_dso3(
        self, rows: list[DSO3PredictRequest]
    ) -> list[dict[str, Any]]:
        return [self.predict_dso3(row) for row in rows]

    def dso1_evaluation(self) -> dict[str, Any]:
        actual = self.dso1.y_test.astype(int).tolist()
        predicted = self.dso1.y_pred.astype(int).tolist()
        probability = (
            self.dso1.y_proba[:, 1].astype(float).tolist()
        )
        points = [
            {
                "index": idx,
                "actual": a,
                "predicted": p,
                "probability": prob,
            }
            for idx, (a, p, prob) in enumerate(
                zip(actual, predicted, probability)
            )
        ]
        metrics = {
            "accuracy": float(
                accuracy_score(actual, predicted)
            ),
            "n_samples": len(actual),
        }
        return {
            "model": "XGBoostClassifier",
            "metrics": metrics,
            "points": points,
        }

    def dso2_evaluation(self) -> dict[str, Any]:
        actual = self.dso2.y_test.astype(float).tolist()
        predicted = self.dso2.y_pred.astype(float).tolist()
        points = [
            {"index": idx, "actual": a, "predicted": p}
            for idx, (a, p) in enumerate(
                zip(actual, predicted)
            )
        ]
        metrics = {
            "r2": float(r2_score(actual, predicted)),
            "mae": float(
                mean_absolute_error(actual, predicted)
            ),
            "rmse": float(
                np.sqrt(mean_squared_error(actual, predicted))
            ),
            "n_samples": len(actual),
        }
        return {
            "model": "PolynomialRegression(degree=2)",
            "metrics": metrics,
            "points": points,
        }

    def dso3_overview(self) -> dict[str, Any]:
        clusters = self.dso3.y_clusters_test.astype(int).tolist()
        anomaly_flags = (
            self.dso3.y_anomaly_test.astype(bool).tolist()
        )
        points = [
            {
                "index": idx,
                "pc1": float(point[0]),
                "pc2": float(point[1]),
                "cluster": int(cluster),
                "is_anomaly": bool(anomaly),
            }
            for idx, (point, cluster, anomaly) in enumerate(
                zip(
                    self.dso3.projection_test,
                    clusters,
                    anomaly_flags,
                )
            )
        ]
        anomaly_rate = float(
            sum(anomaly_flags) / max(len(anomaly_flags), 1)
        )
        summary = {
            "n_samples": len(points),
            "n_clusters": int(len(np.unique(clusters))),
            "n_anomalies": int(sum(anomaly_flags)),
            "anomaly_rate": anomaly_rate,
            "silhouette": float(self.dso3.silhouette),
            "eps": float(self.dso3.eps),
            "pca_kaiser_components": int(
                self.dso3.pca_kaiser_components
            ),
        }
        return {
            "model": "PCA + KMeans + DBSCAN",
            "summary": summary,
            "pca_eigenvalues": [
                float(value) for value in self.dso3.pca_eigenvalues
            ],
            "pca_explained_variance_ratio": [
                float(value)
                for value in self.dso3.pca_explained_variance_ratio
            ],
            "pca_kaiser_components": int(
                self.dso3.pca_kaiser_components
            ),
            "points": points,
        }
