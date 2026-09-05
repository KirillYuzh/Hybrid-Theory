import csv
import hashlib
import io
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "external"
CACHE_DIR = DATA_DIR / "cache"


@dataclass
class LabeledAddress:
    address: str
    label: str
    source: str
    timestamp: str
    category: str = ""


CONFIDENCE_WEIGHTS: dict[str, float] = {
    "ethereum_lists": 0.8,
    "open_source": 0.5,
    "opensanctions": 0.7,
}

DEFAULT_CONFIDENCE = 0.5


def _cached_fetch(url: str, max_age_hours: int = 24, timeout: int = 30) -> str:
    """Fetch URL with filesystem caching."""
    import urllib.request

    cache_key = hashlib.md5(url.encode()).hexdigest()
    cache_path = CACHE_DIR / f"{cache_key}.json"

    if cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < max_age_hours * 3600:
            return cache_path.read_text()

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "kyt-engine/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode()
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(body)
            return body
        except Exception as exc:
            last_exc = exc
            logger.warning("fetch %s attempt %d failed: %s", url, attempt + 1, exc)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch {url} after 3 attempts: {last_exc}")


def _extract_eth_addresses(text: str) -> list[str]:
    """Extract Ethereum addresses from raw text."""
    pattern = re.compile(r"\b0x[0-9a-fA-F]{40}\b")
    return pattern.findall(text)


class OpenSanctionsScraper:
    """OpenSanctions — sanctions, PEP and regulatory data source.

    Fetches CryptoWallet addresses from the OpenSanctions default dataset,
    which includes wallets associated with sanctioned entities, fraud,
    and illicit activity. Source: https://www.opensanctions.org/datasets/default/
    """

    CSV_URL = "https://data.opensanctions.org/artifacts/default/{timestamp}-kfx/targets.simple.csv"

    def __init__(self, timeout: int = 60):
        self.timeout = timeout

    def fetch(self) -> List[LabeledAddress]:
        """Fetch OpenSanctions CryptoWallet data.

        Tries the most recent available timestamp by checking the dataset
        index, then falls back to guessing recent hourly timestamps.
        """
        import urllib.request, re as re_mod
        # Try fetching the index to find the most recent timestamp
        try:
            idx_url = "https://data.opensanctions.org/datasets/latest/default/index.json"
            with urllib.request.urlopen(idx_url, timeout=self.timeout) as resp:
                idx_data = resp.read().decode("utf-8", errors="replace")
            idx = json.loads(idx_data)
            # Extract timestamp from resource URLs
            for resource in idx.get("resources", []):
                full_url = resource.get("url", "")
                m = re_mod.search(r"(\d{14})-kfx", full_url)
                if m:
                    alt_ts = m.group(1)
                    csv_url = self.CSV_URL.format(timestamp=alt_ts)
                    try:
                        with urllib.request.urlopen(
                            csv_url, timeout=self.timeout
                        ) as csv_resp:
                            csv_text = csv_resp.read().decode("utf-8", errors="replace")
                            return self._extract_crypto_addresses(csv_text)
                    except Exception as e:
                        logger.warning(
                            "OpenSanctions CSV fetch failed for %s: %s", alt_ts, e
                        )
        except Exception as e:
            logger.warning("OpenSanctions index fetch failed: %s", e)

        # Fallback: try guessing recent timestamps (current hour + previous hours)
        from datetime import datetime, timezone, timedelta

        now = datetime.now(timezone.utc)
        for hours_ago in range(0, 5):
            ts = (now - timedelta(hours=hours_ago)).strftime("%Y%m%d%H%M%S")
            csv_url = self.CSV_URL.format(timestamp=ts)
            try:
                with urllib.request.urlopen(csv_url, timeout=self.timeout) as csv_resp:
                    csv_text = csv_resp.read().decode("utf-8", errors="replace")
                    return self._extract_crypto_addresses(csv_text)
            except Exception:
                continue

        return []

    @staticmethod
    def _extract_crypto_addresses(csv_text: str) -> List[LabeledAddress]:
        """Extract crypto wallet addresses from OpenSanctions CSV."""
        import csv as csv_mod
        address_pattern = re.compile(r"\b0x[0-9a-fA-F]{40}\b|\bc1[qp][0-9a-z]{39,81}\b")
        addresses: List[LabeledAddress] = []
        lines = csv_text.strip().split("\n")
        if not lines:
            return addresses
        for line in lines[1:]:
            if not line.strip():
                continue
            fields = csv_mod.reader([line]).__next__()
            # Schema: id,name,aliases,birth_date,countries,addresses,identifiers,sanctions,...
            addr_field = fields[6] if len(fields) > 6 else ""
            ident_field = fields[7] if len(fields) > 7 else ""
            # Use identifiers as the address if available, otherwise addresses
            candidate = ident_field.strip() if ident_field else addr_field.strip()
            if not candidate:
                continue
            # Extract all matching addresses from the field
            for match in address_pattern.finditer(candidate):
                addr = match.group(0)
                if addr.startswith("0x"):
                    cat = "sanctions"
                else:
                    cat = "scam"
                addresses.append(
                    LabeledAddress(
                        address=addr.lower(),
                        label="illicit",
                        source="opensanctions",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        category=cat,
                    )
                )
        return addresses


class Scraper:
    """Unified source of external labels with confidence weights and conflict resolution."""

    confidence_weights: dict[str, float] = CONFIDENCE_WEIGHTS

    def __init__(self, sources: Optional[list] = None):
        if sources is None:
            self._sources = [OpenSanctionsScraper()]
        else:
            self._sources = sources

    def fetch_all(self) -> pd.DataFrame:
        """Collects data from all sources and returns merged DataFrame."""
        frames: list[pd.DataFrame] = []
        for src in self._sources:
            try:
                raw = src.fetch()
                if isinstance(raw, pd.DataFrame):
                    frames.append(
                        self._normalize_df(raw, source=src.__class__.__name__.lower())
                    )
                else:
                    rows = [
                        {
                            "address": r.address.lower(),
                            "label": r.label,
                            "source": r.source,
                            "category": r.category,
                            "timestamp": r.timestamp,
                        }
                        for r in raw
                    ]
                    df = pd.DataFrame(rows)
                    frames.append(
                        self._normalize_df(df, source=src.__class__.__name__.lower())
                    )
            except Exception as exc:
                logger.warning("Scraper source %s failed: %s", type(src).__name__, exc)
        if not frames:
            return pd.DataFrame(
                columns=["address", "label", "source", "confidence", "timestamp"]
            )
        return self._merge_sources(frames)

    def _normalize_df(self, raw: pd.DataFrame, source: str) -> pd.DataFrame:
        """Normalize raw data to a common schema."""
        if raw.empty:
            return pd.DataFrame(
                columns=["address", "label", "source", "confidence", "timestamp"]
            )

        df = raw.copy()
        if "address" not in df.columns:
            if "addr" in df.columns:
                df["address"] = df["addr"]
            else:
                df["address"] = df.index.astype(str)
        df["address"] = df["address"].astype(str).str.lower()

        if "label" not in df.columns:
            df["label"] = "illicit"
        if "source" not in df.columns:
            df["source"] = source
        if "timestamp" not in df.columns:
            df["timestamp"] = int(datetime.now(timezone.utc).timestamp())
        if "confidence" not in df.columns:
            df["confidence"] = self.confidence_weights.get(source, DEFAULT_CONFIDENCE)

        df["timestamp"] = df["timestamp"].apply(
            lambda ts: int(datetime.fromisoformat(ts).timestamp())
            if isinstance(ts, str)
            else int(ts)
        )

        return df[["address", "label", "source", "confidence", "timestamp"]]

    def _merge_sources(self, df_list: list[pd.DataFrame]) -> pd.DataFrame:
        """Confidence-weighted merge; label conflicts -> label='REVIEW', confidence=0.5."""
        if not df_list:
            return pd.DataFrame(
                columns=["address", "label", "source", "confidence", "timestamp"]
            )

        combined = pd.concat(df_list, ignore_index=True)
        combined = combined.sort_values("confidence", ascending=False)

        dup_addresses = combined[combined.duplicated(subset=["address"], keep=False)][
            "address"
        ].unique()
        conflicting: set[str] = set()
        for addr in dup_addresses:
            labels = set(combined.loc[combined["address"] == addr, "label"])
            if len(labels) > 1:
                conflicting.add(addr)

        conflict_mask = combined["address"].isin(conflicting)
        combined.loc[conflict_mask, "label"] = "REVIEW"
        combined.loc[conflict_mask, "confidence"] = 0.5

        result = combined.drop_duplicates(
            subset=["address"], keep="first"
        ).reset_index(drop=True)
        return result[["address", "label", "source", "confidence", "timestamp"]]


def run_full_scrape() -> pd.DataFrame:
    """Run all scrapers and save merged results."""
    scrapers = [OpenSanctionsScraper()]
    all_labels: list[LabeledAddress] = []
    for s in scrapers:
        try:
            labels = s.fetch()
            logger.info("Fetched %d labels from %s", len(labels), type(s).__name__)
            all_labels.extend(labels)
        except Exception as e:
            logger.warning("Scraper %s failed: %s", type(s).__name__, e)

    store = ExternalLabelStore()
    df = store.merge(all_labels)
    path = store.save(df)
    logger.info("Saved %d external labels to %s", len(df), path)
    return df


class ExternalLabelStore:
    """Manages the external labels database. Merges labels from multiple sources."""

    PRIORITY = {"ethereum_lists": 2, "open_source": 1, "opensanctions": 3}

    def __init__(self, storage_dir: Optional[Path] = None):
        self._dir = storage_dir or DATA_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    def merge(self, all_labels: list[LabeledAddress]) -> pd.DataFrame:
        if not all_labels:
            return pd.DataFrame(
                columns=["address", "label", "source", "timestamp", "category"]
            )

        rows = []
        for lbl in all_labels:
            rows.append(
                {
                    "address": lbl.address.lower(),
                    "label": lbl.label,
                    "source": lbl.source,
                    "timestamp": lbl.timestamp,
                    "category": lbl.category,
                    "priority": self.PRIORITY.get(lbl.source, 0),
                }
            )

        df = pd.DataFrame(rows)
        df = df.sort_values("priority", ascending=False)
        df = df.drop_duplicates(subset=["address"], keep="first")
        df = df.drop(columns=["priority"])
        df = df.reset_index(drop=True)
        return df

    def save(self, df: pd.DataFrame) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self._dir / f"external_labels_{ts}.parquet"
        df.to_parquet(path, index=False)
        return path

    def load(self) -> pd.DataFrame:
        parquets = sorted(self._dir.glob("external_labels_*.parquet"))
        if not parquets:
            return pd.DataFrame(
                columns=["address", "label", "source", "timestamp", "category"]
            )
        return pd.read_parquet(parquets[-1])

    def get_illicit_addresses(self) -> set[str]:
        df = self.load()
        if df.empty:
            return set()
        return set(df.loc[df["label"] == "illicit", "address"])

    def get_statistics(self) -> dict:
        df = self.load()
        if df.empty:
            return {"total": 0, "by_source": {}, "by_label": {}}
        return {
            "total": len(df),
            "by_source": df["source"].value_counts().to_dict(),
            "by_label": df["label"].value_counts().to_dict(),
        }