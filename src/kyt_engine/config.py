DEFAULT_RISK_ZONE_THRESHOLDS = (0.3, 0.7)
DEFAULT_RISK_ZONE_LABELS = ("GREEN", "YELLOW", "RED")

LIGHTGBM_WEIGHT = 0.5
KSCORE_WEIGHT = 0.2
VAE_WEIGHT = 0.15
EXTERNAL_WEIGHT = 0.15

CONFIDENCE_WEIGHTS: dict[str, float] = {
    "ethereum_lists": 0.8,
    "open_source": 0.5,
}

DEFAULT_CONFIDENCE = 0.3

ACTIVE_LEARNING = {
    "default_top_k": 1000,
    "entropy_threshold": 0.7,
    "kscore_threshold": 0.5,
}

ENSEMBLE = {
    "ae_params": {
        "epochs": 50,
        "batch_size": 128,
        "device": "cpu",
    }
}

DEFAULT = {
    "window_days": (30, 60, 90),
    "default_threshold": 0.7,
    "z_score_max": 3.0,
}

# Default feature column configuration for Spark feature extraction
DEFAULT_FEATURE_COLS = [f"f{i}" for i in range(1, 167)]
DEFAULT_NUMERIC_COLS = ["value", "gas_price", "gas_used"]