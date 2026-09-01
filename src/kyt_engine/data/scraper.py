from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "external"


@dataclass
class LabeledAddress:
    address: str
    label: str
    source: str
    timestamp: str
    category: str = ""


def _fetch_url(url: str, timeout: int = 30) -> str:
    """Fetch raw text from *url* with a standard User-Agent header."""
    req = urllib.request.Request(url, headers={"User-Agent": "kyt-engine/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode()


def _addresses_to_dicts(addresses: list[LabeledAddress]) -> list[dict]:
    return [
        {
            "address": a.address,
            "label": a.label,
            "source": a.source,
            "timestamp": a.timestamp,
            "category": a.category,
        }
        for a in addresses
    ]


class CryptoScamDBScraper:
    """Scrapes CryptoScamDB for known scam addresses."""

    BASE_URL = "https://api.cryptoscamdb.org/v1"

    def fetch(self, max_pages: int = 5) -> list[LabeledAddress]:
        all_addresses: list[LabeledAddress] = []
        now = datetime.now(timezone.utc).isoformat()

        for page in range(1, max_pages + 1):
            url = f"{self.BASE_URL}/addresses?page={page}"
            try:
                data = json.loads(_fetch_url(url))
            except (urllib.error.URLError, json.JSONDecodeError, OSError) as e:
                logger.warning("CryptoScamDB page %d failed: %s", page, e)
                break

            result = data.get("result", [])
            if not result:
                break

            for entry in result:
                addr = entry.get("address", "")
                if not addr:
                    continue
                tag = entry.get("tag", "")
                category = tag.lower() if tag else "scam"
                all_addresses.append(
                    LabeledAddress(
                        address=addr,
                        label="illicit",
                        source="cryptoscamdb",
                        timestamp=now,
                        category=category,
                    )
                )

        logger.info("CryptoScamDB: fetched %d addresses", len(all_addresses))
        return all_addresses


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
                raw = _fetch_url(url)
                data = json.loads(raw)
            except (urllib.error.URLError, json.JSONDecodeError, OSError) as e:
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


class OFACScraper:
    """Scrapes OFAC/US Treasury sanctions list for sanctioned crypto addresses."""

    BASE_URL = "https://data.opensanctions.org/datasets/latest/default"

    def fetch(self) -> list[LabeledAddress]:
        all_addresses: list[LabeledAddress] = []
        now = datetime.now(timezone.utc).isoformat()

        try:
            raw = _fetch_url(f"{self.BASE_URL}/targets.simple.csv", timeout=60)
        except (urllib.error.URLError, OSError) as e:
            logger.warning("OFAC CSV fetch failed: %s", e)
            return all_addresses

        lines = raw.strip().split("\n")
        if not lines:
            return all_addresses

        header = lines[0].split(",")
        try:
            addr_idx = header.index("properties.currencyAddress")
        except ValueError:
            try:
                addr_idx = header.index("cryptoCurrencyAddress")
            except ValueError:
                logger.warning("OFAC CSV: no address column found")
                return all_addresses

        for line in lines[1:]:
            cols = line.split(",")
            if addr_idx >= len(cols):
                continue
            addr = cols[addr_idx].strip().strip('"')
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


class ExternalLabelStore:
    """Manages the external labels database. Merges labels from multiple sources."""

    PRIORITY = {"ofac": 3, "cryptoscamdb": 2, "ethereum-lists": 1}

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


def run_full_scrape(max_pages: int = 3) -> pd.DataFrame:
    """Run all scrapers and save merged results."""
    scrapers = [CryptoScamDBScraper(), OFACScraper(), EthereumListsScraper()]
    all_labels: list[LabeledAddress] = []
    for s in scrapers:
        try:
            if isinstance(s, CryptoScamDBScraper):
                labels = s.fetch(max_pages=max_pages)
            else:
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
