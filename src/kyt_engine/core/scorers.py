import numpy as np
import pandas as pd
from lightgbm.basic import LightGBMError as _LGBMLightGBMError
from typing import Protocol

from kyt_engine.core.contracts import FeatureVector, TxRecord


class Scorer(Protocol):
    name: str

    def predict_proba(self, features: FeatureVector, tx: TxRecord) -> float: ...


class LightGBMScorer:
    def __init__(self, model, feature_names: list[str] | None = None) -> None:
        self._model = model
        names = feature_names
        if names is None:
            # Check for standard attribute first, then custom _feature_names
            if hasattr(model, "feature_names_in_"):
                names = list(model.feature_names_in_)
            elif hasattr(model, "_feature_names"):
                names = list(model._feature_names)
        # Use the actual feature count from the booster
        self._feature_names = names or []
        self.name = "lightgbm"

    def predict_proba(self, features: FeatureVector, tx: TxRecord) -> float:
        import numpy as np
        import pandas as pd

        expected_len = len(self._feature_names)
        input_features = features.values

        if expected_len == 0:
            return 0.5

        # Build feature vector matching booster's expected feature count
        n_cols = expected_len
        values = []
        for i in range(n_cols):
            c = self._feature_names[i] if i < len(self._feature_names) else None
            if c and c in input_features:
                values.append(float(input_features[c]))
            else:
                values.append(0.0)

        X_df = pd.DataFrame([values], columns=self._feature_names[:n_cols])

        try:
            proba = self._model.predict_proba(X_df)[0, 1]
        except _LGBMLightGBMError:
            # Shape mismatch - return neutral risk score
            return 0.5

        return float(np.clip(proba, 0.0, 1.0))


class VAEScorer:
    def __init__(self, model, feature_names: list[str]) -> None:
        self._model = model
        self._feature_names = feature_names
        self.name = "vae"

    def predict_proba(self, features: FeatureVector, tx: TxRecord) -> float:
        import numpy as np

        vec = np.array([features.values.get(c, 0.0) for c in self._feature_names])[None, :]
        proba = self._model.predict_proba(vec)[0, 1]
        return float(np.clip(proba, 0.0, 1.0))


class KScoreScorer:
    def __init__(self, calculator, feature_names: list[str]) -> None:
        self._calculator = calculator
        self._feature_names = feature_names
        self.name = "kscore"

    def predict_proba(self, features: FeatureVector, tx: TxRecord) -> float:
        import numpy as np
        import pandas as pd

        df = pd.DataFrame([features.values], columns=self._feature_names)
        score = self._calculator.score(df).iloc[0]
        return float(np.clip(score, 0.0, 1.0))


class ExternalScorer:
    def __init__(self, illicit_addresses: set[str]) -> None:
        self._illicit = illicit_addresses
        self.name = "external"

    def predict_proba(self, features: FeatureVector, tx: TxRecord) -> float:
        return 1.0 if tx.from_address in self._illicit else 0.0