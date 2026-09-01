from __future__ import annotations

import pyiceberg
from pyiceberg.catalog import load_catalog
from pyiceberg.schema import Schema
from pyiceberg.types import (
    NestedField,
    StringType,
    LongType,
    DoubleType,
    TimestampType,
    BooleanType,
)
import pandas as pd
from datetime import datetime
from typing import Optional


class IcebergStore:
    """Iceberg table management for KYT data with DuckDB catalog."""

    def __init__(self, warehouse_path: str = "warehouse"):
        self._catalog = load_catalog(
            "duckdb",
            **{
                "type": "duckdb",
                "warehouse": warehouse_path,
            }
        )

    def create_namespace(self, namespace: str):
        self._catalog.create_namespace_if_not_exists(namespace)

    def get_or_create_table(self, table_name: str, schema: Schema):
        try:
            return self._catalog.load_table(table_name)
        except Exception:
            return self._catalog.create_table(table_name, schema)

    def upsert(self, table_name: str, df: pd.DataFrame, schema: Schema):
        """Upsert dataframe into Iceberg table (merge on primary key)."""
        table = self.get_or_create_table(table_name, schema)

        if df.empty:
            return

        records = df.to_dict("records")
        with table.new_row_delta() as delta:
            delta.add_rows(records)

    def read(self, table_name: str) -> pd.DataFrame:
        table = self._catalog.load_table(table_name)
        return table.scan().to_pandas()

    def read_as_of(self, table_name: str, timestamp: datetime) -> pd.DataFrame:
        """Time travel: read table as of a specific timestamp."""
        table = self._catalog.load_table(table_name)
        return table.scan(snapshot_id=table.history()[-1].snapshot_id).to_pandas()

    def history(self, table_name: str):
        """Get table history for time travel."""
        table = self._catalog.load_table(table_name)
        return table.history()


EXTERNAL_LABELS_SCHEMA = Schema(
    NestedField(1, "address", StringType(), required=True),
    NestedField(2, "label", StringType(), required=True),
    NestedField(3, "source", StringType(), required=True),
    NestedField(4, "confidence", DoubleType(), required=False),
    NestedField(5, "timestamp", TimestampType(), required=True),
    NestedField(6, "category", StringType(), required=False),
    NestedField(7, "metadata", StringType(), required=False),
    identifier_field_ids=[1]
)

TRANSACTION_FEATURES_SCHEMA = Schema(
    NestedField(1, "tx_id", LongType(), required=True),
    NestedField(2, "time_step", LongType(), required=True),
    NestedField(3, "from_address", StringType(), required=True),
    NestedField(4, "to_address", StringType(), required=True),
    NestedField(5, "value", DoubleType(), required=True),
    NestedField(6, "gas_price", DoubleType(), required=True),
    NestedField(7, "label", LongType(), required=False),
    identifier_field_ids=[1]
)

MODEL_PREDICTIONS_SCHEMA = Schema(
    NestedField(1, "tx_id", LongType(), required=True),
    NestedField(2, "model_version", StringType(), required=True),
    NestedField(3, "risk_score", DoubleType(), required=True),
    NestedField(4, "risk_zone", StringType(), required=True),
    NestedField(5, "triage_level", StringType(), required=True),
    NestedField(6, "components", StringType(), required=False),
    NestedField(7, "timestamp", TimestampType(), required=True),
    identifier_field_ids=[1]
)