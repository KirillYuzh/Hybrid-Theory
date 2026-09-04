from __future__ import annotations

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
from typing import Optional

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
}


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


class EthereumListsScraper:
    """Ethereum Lists — phishing и scam адреса из GitHub репозитория."""

    URLS = [
        "https://raw.githubusercontent.com/ethereum-lists/kservices/main/metadata/ETH/phishing-hosts.json",
        "https://raw.githubusercontent.com/ethereum-lists/kservices/main/metadata/ETH/spam-hosts.json",
    ]

    def fetch(self) -> list[LabeledAddress]:
        all_addresses: list[LabeledAddress] = []
        now = datetime.now(timezone.utc).isoformat()

        for url in self.URLS:
            try:
                raw = _cached_fetch(url, max_age_hours=12, timeout=15)
                data = json.loads(raw)
            except Exception as e:
                logger.warning("EthereumLists fetch failed for %s: %s", url, e)
                continue

            items = data if isinstance(data, list) else data.get("addresses", [])
            for entry in items:
                if isinstance(entry, dict):
                    addr = entry.get("address", entry.get("id", ""))
                elif isinstance(entry, str):
                    addr = entry
                else:
                    continue
                if addr.startswith("0x"):
                    all_addresses.append(
                        LabeledAddress(
                            address=addr,
                            label="illicit",
                            source="ethereum_lists",
                            timestamp=now,
                            category="phishing",
                        )
                    )

        logger.info("EthereumLists: fetched %d addresses", len(all_addresses))
        return all_addresses


class GitHubAddressesScraper:
    """Скрапинг адресов с GitHub ethereum-lists (upd: urls darklist).

    Источник: https://github.com/ethereum-lists/urls
    Содержит known-scam адреса в формате URLs.
    """
    DARKLIST_URL = "https://raw.githubusercontent.com/ethereum-lists/urls/master/urls-darklist.json"

    def fetch(self) -> list[LabeledAddress]:
        all_addresses: list[LabeledAddress] = []
        now = datetime.now(timezone.utc).isoformat()

        try:
            raw = _cached_fetch(self.DARKLIST_URL, max_age_hours=24, timeout=30)
            data = json.loads(raw)
        except Exception as e:
            logger.warning("GitHubAddressesScraper fetch failed: %s", e)
            return all_addresses

        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, str):
                    # URLs-darklist can contain full URLs or just addresses
                    # Extract 0x... addresses
                    import re
                    addrs = re.findall(r"\b0x[0-9a-fA-F]{40}\b", entry)
                    for addr in addrs:
                        all_addresses.append(
                            LabeledAddress(
                                address=addr,
                                label="illicit",
                                source="open_source",
                                timestamp=now,
                                category="scam",
                            )
                        )
                elif isinstance(entry, dict):
                    addr = entry.get("address", entry.get("id", ""))
                    if addr.startswith("0x"):
                        all_addresses.append(
                            LabeledAddress(
                                address=addr.lower(),
                                label="illicit",
                                source="open_source",
                                timestamp=now,
                                category="scam",
                            )
                        )

        logger.info("GitHubAddresses: fetched %d addresses", len(all_addresses))
        return all_addresses


class ExternalLabelStore:
    """Manages the external labels database. Merges labels from multiple sources."""

    PRIORITY = {"ethereum_lists": 2, "open_source": 1}

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


class Scraper:
    """Единый источник внешних меток с весами доверия и разрешением конфликтов."""

    confidence_weights: dict[str, float] = CONFIDENCE_WEIGHTS

    def __init__(self, sources: Optional[list] = None):
        if sources is None:
            self._sources = [EthereumListsScraper(), GitHubAddressesScraper()]
        else:
            self._sources = sources

    def fetch_all(self) -> pd.DataFrame:
        """Собирает данные из всех источников и возвращает объединённый DataFrame."""
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
        """Нормализует сырые данные к единой схеме."""
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
            df["confidence"] = self.confidence_weights.get(source, 0.3)

        df["timestamp"] = df["timestamp"].apply(
            lambda ts: int(datetime.fromisoformat(ts).timestamp())
            if isinstance(ts, str)
            else int(ts)
        )

        return df[["address", "label", "source", "confidence", "timestamp"]]

    def _merge_sources(self, df_list: list[pd.DataFrame]) -> pd.DataFrame:
        """Доверительное слияние источников; конфликты меток → label='REVIEW', confidence=0.5."""
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
    scrapers = [EthereumListsScraper(), GitHubAddressesScraper()]
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
