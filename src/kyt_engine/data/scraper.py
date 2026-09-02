from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

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


def cached_fetch(url: str, max_age_hours: int = 24, timeout: int = 30) -> dict | list | str:
    """Fetch URL with filesystem caching and exponential-backoff retry."""
    cache_key = hashlib.md5(url.encode()).hexdigest()
    cache_path = CACHE_DIR / f"{cache_key}.json"

    if cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < max_age_hours * 3600:
            try:
                return json.loads(cache_path.read_text())
            except json.JSONDecodeError:
                pass

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "kyt-engine/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode()
            # Try to parse as JSON; fall back to raw string
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                data = body
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            if isinstance(data, (dict, list)):
                cache_path.write_text(json.dumps(data))
            return data
        except Exception as exc:
            last_exc = exc
            logger.warning("fetch %s attempt %d failed: %s", url, attempt + 1, exc)
            time.sleep(2 ** attempt)

    raise RuntimeError(f"Failed to fetch {url} after 3 attempts: {last_exc}")


def _fetch_url(url: str, timeout: int = 30) -> str:
    """Fetch raw text from *url* (no caching, for internal helpers)."""
    req = urllib.request.Request(url, headers={"User-Agent": "kyt-engine/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode()


# ---------------------------------------------------------------------------
# GoPlus Token Security
# ---------------------------------------------------------------------------

class GoPlusChecker:
    """Checks token security via GoPlus API.

    Detects honeypots, high tax tokens, hidden mint, etc.
    chain_id: 1=ETH, 56=BSC, 137=Polygon, etc.
    """

    BASE_URL = "https://api.gopluslabs.io/api/v1"

    def check_token(self, contract_address: str, chain_id: int = 1) -> dict:
        """Check a single token for security issues.

        Returns a dict with keys like: is_honeypot, buy_tax, sell_tax,
        cannot_sell_all, hidden_owner, is_open_source, etc.
        """
        url = f"{self.BASE_URL}/token_security/{chain_id}?contract_addresses={contract_address}"
        data = cached_fetch(url, max_age_hours=1)

        result_map = data.get("result", {})
        # GoPlus returns addresses as keys; grab the first (only) entry
        if isinstance(result_map, dict):
            for _addr, info in result_map.items():
                return info if isinstance(info, dict) else {}
        return {}

    def check_batch(self, addresses: list[str], chain_id: int = 1) -> list[dict]:
        """Check multiple tokens (max 100 per request)."""
        results: list[dict] = []
        for chunk_start in range(0, len(addresses), 100):
            chunk = addresses[chunk_start : chunk_start + 100]
            joined = ",".join(chunk)
            url = f"{self.BASE_URL}/token_security/{chain_id}?contract_addresses={joined}"
            data = cached_fetch(url, max_age_hours=1)
            result_map = data.get("result", {})
            if isinstance(result_map, dict):
                for addr in chunk:
                    info = result_map.get(addr, {})
                    results.append(info if isinstance(info, dict) else {})
            else:
                results.extend([{}] * len(chunk))
        return results


# ---------------------------------------------------------------------------
# CryptoScamDB replacement — uses ethereum-lists/scam-db GitHub mirror
# ---------------------------------------------------------------------------

class ScamDBScraper:
    """Fetches known scam/illicit addresses from ethereum-lists scam-db mirror.

    The original CryptoScamDB API is dead (502). This uses the community
    maintained GitHub mirror of scam address lists.
    """

    REPOS = [
        # ethereum-lists maintains curated scam-address lists
        "https://raw.githubusercontent.com/ethereum-lists/scam-db/main/data/addresses.json",
        # Alt: transaction revert / phishing list
        "https://raw.githubusercontent.com/ethereum-lists/master/addresses/ETH/0x0000000000000000000000000000000000000000.json",
    ]

    def fetch(self) -> list[LabeledAddress]:
        all_addresses: list[LabeledAddress] = []
        now = datetime.now(timezone.utc).isoformat()

        for url in self.REPOS:
            try:
                data = cached_fetch(url, max_age_hours=12)
            except Exception as e:
                logger.warning("ScamDB fetch failed for %s: %s", url, e)
                continue

            items: list[dict] = []
            if isinstance(data, list):
                items = data if all(isinstance(x, dict) for x in data) else []
            elif isinstance(data, dict):
                items = data.get("result", data.get("addresses", []))
                if not items:
                    # Might be {address: {tag: ...}} format
                    for addr, info in data.items():
                        if isinstance(info, dict) and addr != "result":
                            items.append({"address": addr, **info})

            for entry in items:
                if not isinstance(entry, dict):
                    continue
                addr = entry.get("address", "")
                if not addr or not addr.startswith("0x"):
                    continue
                tag = entry.get("tag", entry.get("category", ""))
                all_addresses.append(
                    LabeledAddress(
                        address=addr,
                        label="illicit",
                        source="scamdb",
                        timestamp=now,
                        category=str(tag).lower() if tag else "scam",
                    )
                )

        logger.info("ScamDB: fetched %d addresses", len(all_addresses))
        return all_addresses


# ---------------------------------------------------------------------------
# Ethereum-lists phishing lists
# ---------------------------------------------------------------------------

class EthereumListsScraper:
    """Scrapes ethereum-lists GitHub for phishing/scam address blacklists."""

    REPOS = [
        "https://raw.githubusercontent.com/ethereum-lists/master/addresses/ETH/tokenAddressess/0xdAC17F958D2ee523a2206206994597C13D831ec7.json",
    ]

    def fetch(self) -> list[LabeledAddress]:
        all_addresses: list[LabeledAddress] = []
        now = datetime.now(timezone.utc).isoformat()

        for url in self.REPOS:
            try:
                data = cached_fetch(url, max_age_hours=12)
            except Exception as e:
                logger.warning("EthereumLists fetch failed for %s: %s", url, e)
                continue

            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = data.get("tokens", data.get("addresses", [data]))
            else:
                continue

            for entry in items:
                if not isinstance(entry, dict):
                    continue
                addr = entry.get("address", "")
                if not addr:
                    continue
                all_addresses.append(
                    LabeledAddress(
                        address=addr,
                        label="illicit",
                        source="ethereum-lists",
                        timestamp=now,
                        category="phishing",
                    )
                )

        logger.info("EthereumLists: fetched %d addresses", len(all_addresses))
        return all_addresses


# ---------------------------------------------------------------------------
# OFAC / OpenSanctions
# ---------------------------------------------------------------------------

class OFACScraper:
    """Scrapes OFAC/US Treasury sanctions list for sanctioned crypto addresses."""

    BASE_URL = "https://data.opensanctions.org/datasets/latest/default"

    def fetch(self) -> list[LabeledAddress]:
        all_addresses: list[LabeledAddress] = []
        now = datetime.now(timezone.utc).isoformat()

        try:
            raw = cached_fetch(f"{self.BASE_URL}/targets.simple.csv", max_age_hours=6, timeout=60)
        except Exception as e:
            logger.warning("OFAC CSV fetch failed: %s", e)
            return all_addresses

        if not isinstance(raw, str):
            raw = str(raw)

        reader = csv.reader(io.StringIO(raw))
        rows = list(reader)
        if not rows:
            return all_addresses

        header = rows[0]
        addr_idx = None
        for candidate in ("properties.currencyAddress", "cryptoCurrencyAddress"):
            if candidate in header:
                addr_idx = header.index(candidate)
                break

        if addr_idx is None:
            logger.warning("OFAC CSV: no address column found in %s", header[:10])
            return all_addresses

        for cols in rows[1:]:
            if addr_idx >= len(cols):
                continue
            addr = cols[addr_idx].strip()
            if not addr or not addr.startswith("0x"):
                continue
            all_addresses.append(
                LabeledAddress(
                    address=addr,
                    label="illicit",
                    source="ofac",
                    timestamp=now,
                    category="sanctions",
                )
            )

        logger.info("OFAC: fetched %d sanctioned addresses", len(all_addresses))
        return all_addresses


# ---------------------------------------------------------------------------
# External Label Store
# ---------------------------------------------------------------------------

class ExternalLabelStore:
    """Manages the external labels database. Merges labels from multiple sources."""

    PRIORITY = {"ofac": 3, "scamdb": 2, "ethereum-lists": 1}

    def __init__(self, storage_dir: Path | None = None):
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


# ---------------------------------------------------------------------------
# Full scrape orchestrator
# ---------------------------------------------------------------------------

def run_full_scrape(max_pages: int = 3) -> pd.DataFrame:
    """Run all scrapers and save merged results."""
    scrapers = [ScamDBScraper(), OFACScraper(), EthereumListsScraper()]
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


# ---------------------------------------------------------------------------
# Unified Scraper — trust-weighted merge + conflict resolution
# ---------------------------------------------------------------------------

class Scraper:
    """
    Единый источник внешних меток с весами доверия и разрешением конфликтов.

    Агрегирует данные из нескольких источников (OFAC, CryptoScamDB, ethereum-lists),
    нормализует их к единой схеме и выполняет доверительное слияние. Если источники
    противоречат друг другу по одному адресу — эскалирует на ручную проверку.
    """

    confidence_weights: dict[str, float] = {
        "ofac": 1.0,
        "cryptoscamdb": 0.7,
        "github": 0.3,
    }

    def __init__(self, sources: list | None = None):
        """sources — список объектов с методом fetch() -> DataFrame/IPython list[LabeledAddress].

        По умолчанию использует реальные HTTP-источники. Для офлайн-тестов можно
        внедрить мок-объекты с тем же методом fetch().
        """
        if sources is None:
            self._sources = [ScamDBScraper(), OFACScraper(), EthereumListsScraper()]
        else:
            self._sources = sources

    def fetch_all(self) -> pd.DataFrame:
        """Собирает данные из всех источников и возвращает объединённый DataFrame."""
        frames: list[pd.DataFrame] = []
        for src in self._sources:
            try:
                raw = src.fetch()
                # Поддержка объектов, возвращающих list[LabeledAddress], и прямых DataFrame
                if isinstance(raw, pd.DataFrame):
                    frames.append(self.normalize(raw, source=src.__class__.__name__.lower()))
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
                    frames.append(self.normalize(df, source=src.__class__.__name__.lower()))
            except Exception as exc:
                logger.warning("Scraper source %s failed: %s", type(src).__name__, exc)
        if not frames:
            return pd.DataFrame(columns=["address", "label", "source", "confidence", "timestamp"])
        return self.merge_sources(frames)

    def normalize(self, raw: pd.DataFrame, source: str) -> pd.DataFrame:
        """Нормализует сырые данные к единой схеме.

        Выходные колонки: (address, label, source, confidence, timestamp)
        confidence — вес доверия источника; timestamp — Unix-время (int).
        """
        if raw.empty:
            return pd.DataFrame(columns=["address", "label", "source", "confidence", "timestamp"])

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
            df["confidence"] = self.confidence_weights.get(source, self.confidence_weights.get(df["source"].iloc[0], 0.3))

        # timestamp → Unix int
        df["timestamp"] = df["timestamp"].apply(
            lambda ts: int(datetime.fromisoformat(ts).timestamp())
            if isinstance(ts, str) else int(ts)
        )

        return df[["address", "label", "source", "confidence", "timestamp"]]

    def merge_sources(self, df_list: list[pd.DataFrame]) -> pd.DataFrame:
        """Доверительное слияние источников; конфликты меток → label='REVIEW', confidence=0.5."""
        if not df_list:
            return pd.DataFrame(columns=["address", "label", "source", "confidence", "timestamp"])

        combined = pd.concat(df_list, ignore_index=True)
        combined = combined.sort_values("confidence", ascending=False)

        # Определяем конфликт: один адрес с разными label от разных источников
        dup_addresses = combined[combined.duplicated(subset=["address"], keep=False)]["address"].unique()
        conflicting: set[str] = set()
        for addr in dup_addresses:
            labels = set(combined.loc[combined["address"] == addr, "label"])
            if len(labels) > 1:
                conflicting.add(addr)

        # Эскалация конфликтов
        conflict_mask = combined["address"].isin(conflicting)
        combined.loc[conflict_mask, "label"] = "REVIEW"
        combined.loc[conflict_mask, "confidence"] = 0.5

        result = combined.drop_duplicates(subset=["address"], keep="first").reset_index(drop=True)
        return result[["address", "label", "source", "confidence", "timestamp"]]
