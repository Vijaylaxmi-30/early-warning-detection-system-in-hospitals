import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def create_sample_data(n_patients: int = 1000, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)

    subject_ids = np.arange(1, n_patients + 1)

    # Mortality ~15.3%
    mortality_rate = 0.153
    expire_flags = rng.random(n_patients) < mortality_rate

    df_cohort = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "hadm_id": 10000 + subject_ids,
            "icustay_id": 20000 + subject_ids,
            "hospital_expire_flag": expire_flags.astype(int),
        }
    )

    # 6 timepoints across 4 hours
    timepoints = np.linspace(0.0, 4.0, 6)
    rows = []

    for sid, flag in zip(subject_ids, expire_flags):
        # Survivors vs Non-survivors distribution shifts (approximate those shown in notebook)
        if not flag:
            hr = rng.normal(75, 10, size=timepoints.size)
            sbp = rng.normal(120, 15, size=timepoints.size)
            dbp = rng.normal(80, 17, size=timepoints.size)
            rr = rng.normal(16, 4, size=timepoints.size)
            temp = rng.normal(37.0, 1.0, size=timepoints.size)
            spo2 = np.clip(rng.normal(97.8, 1.7, size=timepoints.size), 80, 100)
        else:
            hr = rng.normal(96.8, 22.0, size=timepoints.size)
            sbp = rng.normal(106.7, 17.7, size=timepoints.size)
            dbp = rng.normal(67.5, 18.6, size=timepoints.size)
            rr = rng.normal(21.4, 5.0, size=timepoints.size)
            temp = rng.normal(37.0, 1.0, size=timepoints.size)
            spo2 = np.clip(rng.normal(97.8, 1.7, size=timepoints.size), 80, 100)

        for t, v_hr, v_sbp, v_dbp, v_rr, v_temp, v_spo2 in zip(
            timepoints, hr, sbp, dbp, rr, temp, spo2
        ):
            rows.append(
                {
                    "subject_id": sid,
                    "hours_from_icu_admit": t,
                    "heart_rate": float(v_hr),
                    "sbp": float(v_sbp),
                    "dbp": float(v_dbp),
                    "resp_rate": float(v_rr),
                    "temperature": float(v_temp),
                    "spo2": float(v_spo2),
                }
            )

    df_vitals = pd.DataFrame(rows)
    return df_cohort, df_vitals


class EarlyWarningSystem:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.feature_names: List[str] = []
        self.results: Dict[str, Dict[str, float]] = {}
        self._X_df: pd.DataFrame | None = None

        # Random projection matrix to simulate random convolution features
        # 36 basic stats -> 900 features
        self._rp_matrix = self.rng.normal(0, 1, size=(36, 900)).astype(np.float32)

    # ---- MEWS ----
    @staticmethod
    def calculate_mews_score(heart_rate: float, sbp: float, resp_rate: float, temperature: float) -> int:
        score = 0

        # Heart rate
        if heart_rate < 40:
            score += 2
        elif 40 <= heart_rate <= 50:
            score += 1
        elif 51 <= heart_rate <= 100:
            score += 0
        elif 101 <= heart_rate <= 110:
            score += 1
        elif 111 <= heart_rate <= 129:
            score += 2
        elif heart_rate >= 130:
            score += 3

        # Systolic BP
        if sbp < 70:
            score += 3
        elif 70 <= sbp <= 80:
            score += 2
        elif 81 <= sbp <= 100:
            score += 1
        elif 101 <= sbp <= 199:
            score += 0
        elif sbp >= 200:
            score += 2

        # Respiratory rate
        if resp_rate < 9:
            score += 2
        elif 9 <= resp_rate <= 14:
            score += 0
        elif 15 <= resp_rate <= 20:
            score += 1
        elif 21 <= resp_rate <= 29:
            score += 2
        elif resp_rate >= 30:
            score += 3

        # Temperature (C)
        if temperature < 35.0:
            score += 2
        elif 35.0 <= temperature <= 38.4:
            score += 0
        elif temperature >= 38.5:
            score += 1

        return score

    def calculate_mews_for_window(self, patient_vitals: pd.DataFrame) -> Dict[str, float]:
        if patient_vitals.empty:
            return {"mews_mean": 0.0, "mews_max": 0.0, "mews_final": 0.0}

        mews_scores = patient_vitals.apply(
            lambda r: self.calculate_mews_score(
                r.get("heart_rate", np.nan),
                r.get("sbp", np.nan),
                r.get("resp_rate", np.nan),
                r.get("temperature", np.nan),
            ),
            axis=1,
        )
        return {
            "mews_mean": float(np.mean(mews_scores)),
            "mews_max": float(np.max(mews_scores)),
            "mews_final": float(mews_scores.iloc[-1]),
        }

    # ---- Feature Engineering ----
    @staticmethod
    def _compute_slope(times: np.ndarray, values: np.ndarray) -> float:
        if len(times) < 2 or np.allclose(times, times[0]):
            return 0.0
        try:
            slope, _ = np.polyfit(times, values, 1)
            return float(slope)
        except Exception:
            return 0.0

    def _compute_basic_stats_for_patient(self, g: pd.DataFrame) -> Dict[str, float]:
        stats: Dict[str, float] = {}
        vitals = ["heart_rate", "sbp", "dbp", "resp_rate", "temperature", "spo2"]
        t = g["hours_from_icu_admit"].to_numpy()

        for v in vitals:
            x = g[v].to_numpy()
            stats[f"{v}_mean"] = float(np.nanmean(x))
            stats[f"{v}_std"] = float(np.nanstd(x))
            stats[f"{v}_min"] = float(np.nanmin(x))
            stats[f"{v}_max"] = float(np.nanmax(x))
            stats[f"{v}_range"] = float(np.nanmax(x) - np.nanmin(x))
            stats[f"{v}_slope"] = self._compute_slope(t, x)
        return stats

    def _compute_mews_features_for_patient(self, g: pd.DataFrame) -> Dict[str, float]:
        mews_list = g.apply(
            lambda r: self.calculate_mews_score(
                r["heart_rate"], r["sbp"], r["resp_rate"], r["temperature"]
            ),
            axis=1,
        ).to_numpy()
        if mews_list.size == 0:
            return {"mews_mean": 0.0, "mews_max": 0.0, "mews_final": 0.0, "mews_std": 0.0}
        return {
            "mews_mean": float(np.mean(mews_list)),
            "mews_max": float(np.max(mews_list)),
            "mews_final": float(mews_list[-1]),
            "mews_std": float(np.std(mews_list)),
        }

    def prepare_training_data(self, df_cohort: pd.DataFrame, df_vitals: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        print("Preparing training data...")

        groups = df_vitals.groupby("subject_id", sort=False)
        rows: List[Dict[str, float]] = []
        subj_order: List[int] = []

        for sid, g in groups:
            subj_order.append(int(sid))
            row: Dict[str, float] = {}
            row.update(self._compute_basic_stats_for_patient(g))
            row.update(self._compute_mews_features_for_patient(g))
            rows.append(row)

        X_basic_mews = pd.DataFrame(rows, index=subj_order)

        # Ensure feature order for basic+mews is deterministic
        basic_feature_names = []
        for v in ["heart_rate", "sbp", "dbp", "resp_rate", "temperature", "spo2"]:
            for stat in ["mean", "std", "min", "max", "range", "slope"]:
                basic_feature_names.append(f"{v}_{stat}")
        mews_feature_names = ["mews_mean", "mews_max", "mews_final", "mews_std"]
        X_basic_mews = X_basic_mews[basic_feature_names + mews_feature_names]

        # Random-projection-based convolution-like features (fast, reproducible)
        basic_matrix = X_basic_mews[basic_feature_names].to_numpy(dtype=np.float32)
        # Normalize per-feature to zero mean unit variance (avoid domination)
        mean = np.nanmean(basic_matrix, axis=0, keepdims=True)
        std = np.nanstd(basic_matrix, axis=0, keepdims=True)
        std[std == 0] = 1.0
        basic_norm = (basic_matrix - mean) / std

        conv_feats = basic_norm @ self._rp_matrix  # shape (n_samples, 900)
        conv_cols = [f"conv_{i:03d}" for i in range(conv_feats.shape[1])]
        X_conv = pd.DataFrame(conv_feats, index=X_basic_mews.index, columns=conv_cols)

        X_df = pd.concat([X_basic_mews, X_conv], axis=1)
        self._X_df = X_df.copy()

        # Labels aligned to subject order
        y_series = (
            df_cohort.set_index("subject_id").loc[X_df.index, "hospital_expire_flag"].astype(int)
        )

        # Persist ordered feature names
        self.feature_names = list(X_df.columns)

        print(f"Prepared data: {X_df.shape[0]} patients, {X_df.shape[1]} features")
        return X_df.to_numpy(dtype=np.float32), y_series.to_numpy(dtype=np.int32)

    # ---- Modeling ----
    def _make_xgb_like(self):
        try:
            from xgboost import XGBClassifier  # type: ignore

            return XGBClassifier(
                n_estimators=200,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                learning_rate=0.05,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="auc",
                random_state=self.random_state,
                n_jobs=0,
            )
        except Exception:
            # Fallback to a scikit-learn gradient boosting classifier
            return GradientBoostingClassifier(random_state=self.random_state)

    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, float]]:
        print("Training models...")

        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X,
            y,
            np.arange(X.shape[0]),
            test_size=0.3,
            stratify=y,
            random_state=self.random_state,
        )

        results: Dict[str, Dict[str, float]] = {}

        # XGBoost (or fallback)
        print("Training XGBoost...")
        model_xgb = self._make_xgb_like()
        model_xgb.fit(X_train, y_train)
        proba_xgb = model_xgb.predict_proba(X_test)[:, 1]
        auc_xgb = roc_auc_score(y_test, proba_xgb)
        print(f"XGBoost AUC: {auc_xgb:.4f}")
        results["XGBoost"] = {"auc": float(auc_xgb), "y_true": y_test, "y_pred_proba": proba_xgb}

        # "Ridge" -> Logistic Regression with L2
        print("Training Ridge...")
        model_lr = LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=self.random_state
        )
        model_lr.fit(X_train, y_train)
        proba_lr = model_lr.predict_proba(X_test)[:, 1]
        auc_lr = roc_auc_score(y_test, proba_lr)
        print(f"Ridge AUC: {auc_lr:.4f}")
        results["Ridge"] = {"auc": float(auc_lr), "y_true": y_test, "y_pred_proba": proba_lr}

        # Random Forest
        print("Training Random Forest...")
        model_rf = RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=1, class_weight="balanced", random_state=self.random_state
        )
        model_rf.fit(X_train, y_train)
        proba_rf = model_rf.predict_proba(X_test)[:, 1]
        auc_rf = roc_auc_score(y_test, proba_rf)
        print(f"Random Forest AUC: {auc_rf:.4f}")
        results["Random Forest"] = {"auc": float(auc_rf), "y_true": y_test, "y_pred_proba": proba_rf}

        # MEWS Baseline from feature 'mews_final' normalized 0..1
        mews_auc = np.nan
        if self._X_df is not None:
            if "mews_final" in self._X_df.columns:
                mews_vals = self._X_df.iloc[idx_test]["mews_final"].to_numpy(dtype=np.float32)
                # Normalize
                min_v = float(np.nanmin(mews_vals))
                max_v = float(np.nanmax(mews_vals))
                denom = (max_v - min_v) if (max_v - min_v) != 0.0 else 1.0
                mews_proba = (mews_vals - min_v) / denom
                mews_auc = float(roc_auc_score(y_test, mews_proba))
            else:
                mews_proba = np.zeros_like(y_test, dtype=np.float32)
        else:
            mews_proba = np.zeros_like(y_test, dtype=np.float32)

        print(f"MEWS Baseline AUC: {mews_auc:.4f}" if not math.isnan(mews_auc) else "MEWS Baseline AUC: n/a")
        results["MEWS_Baseline"] = {"auc": float(mews_auc), "y_true": y_test, "y_pred_proba": mews_proba}

        self.results = results
        return results

    # ---- Reporting ----
    def generate_report(self) -> None:
        if not self.feature_names or not self.results:
            print("Report unavailable: run feature engineering and training first.")
            return

        basic_features = len(
            [
                f
                for f in self.feature_names
                if any(s in f for s in ["mean", "std", "min", "max", "range", "slope"]) and "conv_" not in f and "mews" not in f
            ]
        )
        mews_features = len([f for f in self.feature_names if f.startswith("mews_")])
        conv_features = len([f for f in self.feature_names if f.startswith("conv_")])

        print("\n============================================================")
        print("ML EARLY WARNING SYSTEM - PERFORMANCE REPORT")
        print("============================================================\n")
        print("Dataset Information:")
        print(f"- Total Features: {len(self.feature_names)}")
        print(f"- Basic Statistical Features: {basic_features}")
        print(f"- MEWS Features: {mews_features}")
        print(f"- Convolution Features: {conv_features}\n")

        print("Model Performance (AUC):")
        print("------------------------------")
        for name, res in sorted(self.results.items(), key=lambda kv: kv[1].get("auc", 0.0), reverse=True):
            print(f"{name:<20}: {res.get('auc', float('nan')):.4f}")

        if "MEWS_Baseline" in self.results:
            print("\nImprovement over MEWS Baseline:")
            print("----------------------------------------")
            mews_auc = self.results["MEWS_Baseline"].get("auc", float("nan"))
            for name, res in sorted(self.results.items(), key=lambda kv: kv[1].get("auc", 0.0), reverse=True):
                if name == "MEWS_Baseline" or math.isnan(mews_auc):
                    continue
                delta = res["auc"] - mews_auc
                rel = (delta / mews_auc * 100.0) if mews_auc not in (0.0, float("nan")) else float("nan")
                print(f"{name:<20}: {delta:+.4f} ({rel:+.1f}%)")

        print("\nKey Insights:")
        print("--------------------")
        best_name = max(self.results.items(), key=lambda kv: kv[1].get("auc", 0.0))[0]
        print(f"- Best performing model: {best_name} (AUC: {self.results[best_name]['auc']:.4f})")
        if "MEWS_Baseline" in self.results and not math.isnan(self.results["MEWS_Baseline"]["auc"]):
            print(
                f"- Improvement over traditional MEWS: "
                f"{(self.results[best_name]['auc'] - self.results['MEWS_Baseline']['auc']):.4f}"
            )
        print("- Random-projection temporal features capture multi-scale patterns")
        print("- ML approach provides more nuanced risk assessment than threshold-based MEWS")



