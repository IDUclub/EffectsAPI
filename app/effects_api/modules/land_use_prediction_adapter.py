import logging
from typing import Optional, Sequence
import geopandas as gpd
import numpy as np
import pandas as pd

from blocksnet.analysis.land_use.prediction import SpatialClassifier

from app.effects_api.constants.const import PRED_VALUE_RU, PROB_COLS_EN_TO_RU

logger = logging.getLogger(__name__)


class LandUsePredictorAdapter:
    """Adapter around SpatialClassifier to produce tidy per-block predictions.

    Responsibilities:
    - Normalize CRS to EPSG:3857 if needed (classifier commonly expects planar metric CRS).
    - Run prediction once and return a DataFrame keyed by block_id.
    - Keep the set of probability columns explicit and stable.
    """

    # какие вероятностные колонки ожидаем от модели
    DEFAULT_PROB_COLUMNS: Sequence[str] = (
        "prob_urban",
        "prob_non_urban",
        "prob_industrial",
    )

    def __init__(
        self,
        classifier: Optional[SpatialClassifier] = None,
        prob_columns: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize adapter with a classifier and explicit prob column names."""
        self._clf = classifier or SpatialClassifier.default()
        self._prob_columns = tuple(prob_columns or self.DEFAULT_PROB_COLUMNS)

    def _ensure_block_index(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Make sure GeoDataFrame is indexed by integer block_id."""
        if "block_id" in gdf.columns:
            gdf["block_id"] = gdf["block_id"].astype(int)
            if gdf.index.name == "block_id":
                gdf = gdf.reset_index(drop=True)
            gdf = (
                gdf.drop_duplicates(subset="block_id", keep="last")
                .set_index("block_id")
                .sort_index()
            )
        else:
            gdf.index = gdf.index.astype(int)
            gdf = gdf[~gdf.index.duplicated(keep="last")].sort_index()
        gdf.index.name = "block_id"
        return gdf

    def _normalize_pred_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert Enums/objects to JSON-safe primitives and enforce dtypes."""

        if "category" in df.columns:
            df = df.drop(columns=["category"])

        if "pred_name" in df.columns:
            df["pred_name"] = df["pred_name"].apply(
                lambda x: getattr(x, "name", x) if x is not None else None
            ).astype("string")

        for c in ("prob_urban", "prob_non_urban", "prob_industrial"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

        return df

    def predict(self, gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """Run land-use prediction on blocks and return a tidy DataFrame.

        Returns DataFrame indexed by block_id with:
          - pred_name (str)
          - prob_urban, prob_non_urban, prob_industrial (float)
        """
        if self._clf is None:
            raise RuntimeError("LandUsePredictorAdapter: classifier is not provided")

        logger.info("Running land-use prediction on base blocks")

        blocks = self._ensure_block_index(gdf)

        y_pred_raw = self._clf.predict(blocks)
        proba = None
        classes = getattr(self._clf, "classes_", None)

        if hasattr(self._clf, "predict_proba"):
            try:
                proba = self._clf.predict_proba(blocks)
            except Exception as _:
                proba = None

        if isinstance(y_pred_raw, pd.Series):
            y_pred = y_pred_raw.reindex(blocks.index)
        elif isinstance(y_pred_raw, pd.DataFrame):
            first_col = y_pred_raw.columns[0] if len(y_pred_raw.columns) else None
            y_pred = y_pred_raw[first_col].reindex(blocks.index) if first_col else pd.Series(index=blocks.index,
                                                                                             dtype="object")
        else:
            y_pred = pd.Series(y_pred_raw, index=blocks.index)

        pred_name = y_pred.apply(
            lambda x: str(getattr(x, "name", x)) if x is not None else None
        ).astype("string")

        if isinstance(proba, np.ndarray) and proba.ndim == 2 and classes is not None:
            proba_df = self._build_proba_df(proba, classes, blocks.index, self._prob_columns)
        else:
            proba_df = pd.DataFrame(index=blocks.index, columns=self._prob_columns, dtype=float)
            for c in self._prob_columns:
                proba_df[c] = np.nan

        out = pd.DataFrame({"pred_name": pred_name}, index=blocks.index).join(proba_df)

        out = self._normalize_pred_columns(out)
        out.index.name = "block_id"

        logger.info("Land-use prediction finished, rows=%d", len(out))
        return out.sort_index()

    def _predict_land_use_for_blocks(self, after_blocks: gpd.GeoDataFrame) -> pd.DataFrame:
        """Run predictor on after_blocks and return normalized predictions."""
        logger.info("Running land-use prediction on base blocks")
        blocks = self._ensure_block_index(after_blocks.copy())
        preds = self.predict(blocks)
        if not isinstance(preds, (pd.DataFrame, gpd.GeoDataFrame)):
            raise ValueError("Predictor must return a DataFrame aligned by block_id")

        preds = preds.copy()
        if preds.index.name != "block_id":
            if "block_id" in preds.columns:
                preds["block_id"] = preds["block_id"].astype(int)
                preds = preds.set_index("block_id")
            else:
                raise ValueError("Predictions must be indexed by 'block_id'")

        preds = self._normalize_pred_columns(preds)

        need_cols = ["pred_name", "prob_urban", "prob_non_urban", "prob_industrial"]
        for c in need_cols:
            if c not in preds.columns:
                preds[c] = np.nan
        preds = preds[need_cols].sort_index()

        logger.info("Land-use prediction finished, rows=%d", len(preds))
        return preds

    def _normalize_class_label(self, x) -> str:
        """Map raw class label to canonical name."""
        if x is None:
            return None
        s = str(getattr(x, "name", x)).strip().lower()
        if "indust" in s:
            return "industrial"
        if "non" in s and "urban" in s:
            return "non_urban"
        if "urb" in s:
            return "urban"
        return s

    def _build_proba_df(
            self,
            proba: np.ndarray,
            classes: Sequence,
            index: pd.Index,
            target_prob_cols: Sequence[str] = ("prob_urban", "prob_non_urban", "prob_industrial"),
    ) -> pd.DataFrame:
        """Build probability DataFrame aligned by block_id with target column names."""
        canonical = [self._normalize_class_label(c) for c in classes]
        temp_cols = [f"prob__{c}" for c in canonical]
        proba_df = pd.DataFrame(proba, index=index, columns=temp_cols)

        out = pd.DataFrame(index=index)
        mapping = {
            "urban": "prob_urban",
            "non_urban": "prob_non_urban",
            "industrial": "prob_industrial",
        }
        for cls_name, temp_col in zip(canonical, temp_cols):
            tgt = mapping.get(cls_name)
            if tgt:
                out[tgt] = proba_df[temp_col]
        for c in target_prob_cols:
            if c not in out.columns:
                out[c] = np.nan
        return out[target_prob_cols]