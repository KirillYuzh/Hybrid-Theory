from __future__ import annotations
import json
import time
import logging
from typing import Iterator, Optional
from dataclasses import dataclass, asdict
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import random

logger = logging.getLogger(__name__)

@dataclass
class RawTransaction:
    tx_id: str
    block_height: int
    timestamp: int
    from_address: str
    to_address: str
    value: float
    gas_price: float
    gas_used: int
    input_data: bytes
    ingestion_ts: int

    def to_dict(self) -> dict:
        return {
            "tx_id": self.tx_id,
            "block_height": self.block_height,
            "timestamp": self.timestamp,
            "from_address": self.from_address,
            "to_address": self.to_address,
            "value": self.value,
            "gas_price": self.gas_price,
            "gas_used": self.gas_used,
            "input_data": self.input_data.hex() if isinstance(self.input_data, bytes) else self.input_data,
            "ingestion_ts": self.ingestion_ts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RawTransaction":
        input_data = d.get("input_data", b"")
        if isinstance(input_data, str):
            input_data = bytes.fromhex(input_data)
        return cls(
            tx_id=d["tx_id"],
            block_height=int(d["block_height"]),
            timestamp=int(d["timestamp"]),
            from_address=d["from_address"],
            to_address=d["to_address"],
            value=float(d["value"]),
            gas_price=float(d["gas_price"]),
            gas_used=int(d["gas_used"]),
            input_data=input_data,
            ingestion_ts=int(d.get("ingestion_ts", int(time.time()))),
        )


class BlockchainIngestor:
    """Collects transactions from blockchain RPC and sends to Kafka."""

    def __init__(
        self,
        kafka_bootstrap_servers: str = "localhost:9092",
        topic: str = "raw_txs",
        rpc_url: Optional[str] = None,
        batch_size: int = 100,
    ):
        self._producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            acks="all",
            retries=3,
            retry_backoff_ms=1000,
        )
        self._topic = topic
        self._rpc_url = rpc_url
        self._batch_size = batch_size
        self._running = False

    def stream_blocks(self, start_block: int = 0) -> Iterator[RawTransaction]:
        """Generator that yields transactions from blockchain.

        In production: connects to Ethereum/Bitcoin RPC.
        In demo mode: generates synthetic transactions.
        """
        self._running = True
        block = start_block

        while self._running:
            try:
                # Production: fetch from RPC
                if self._rpc_url:
                    txs = self._fetch_block_from_rpc(block)
                else:
                    # Demo mode: generate synthetic data
                    txs = self._generate_synthetic_block(block)

                for tx in txs:
                    yield tx
                    # Send to Kafka
                    self._producer.send(
                        self._topic,
                        key=tx.tx_id,
                        value=tx.to_dict(),
                    )
                    if random.random() < 0.01:  # Batch flush
                        self._producer.flush()

                block += 1

            except KafkaError as e:
                logger.error(f"Kafka error at block {block}: {e}")
                time.sleep(5)
            except Exception as e:
                logger.error(f"Error at block {block}: {e}")
                time.sleep(1)

    def _fetch_block_from_rpc(self, block_height: int) -> list[RawTransaction]:
        """Fetch block from Ethereum RPC."""
        import urllib.request
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_getBlockByNumber",
            "params": [hex(block_height), True],
            "id": 1,
        }
        req = urllib.request.Request(
            self._rpc_url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())["result"]
            if not result:
                return []
            txs = []
            for tx in result["transactions"]:
                txs.append(
                    RawTransaction(
                        tx_id=tx["hash"],
                        block_height=block_height,
                        timestamp=int(int(result["timestamp"], 16)),
                        from_address=tx["from"],
                        to_address=tx.get("to", ""),
                        value=float(int(tx["value"], 16)) / 1e18,
                        gas_price=float(int(tx["gasPrice"], 16)) / 1e9,
                        gas_used=int(tx["gas"], 16),
                        input_data=bytes.fromhex(tx["input"][2:]) if tx["input"] != "0x" else b"",
                        ingestion_ts=int(time.time()),
                    )
                )
            return txs

    def _generate_synthetic_block(self, block_height: int) -> list[RawTransaction]:
        """Generate synthetic transactions for demo."""
        n_txs = random.randint(50, 200)
        txs = []
        base_time = int(time.time()) - (1000000 - block_height) * 13

        for i in range(n_txs):
            from_addr = f"0x{random.randint(0, 0xFFFFFF):024x}"
            to_addr = f"0x{random.randint(0, 0xFFFFFF):024x}"
            is_illicit = random.random() < 0.05

            value = random.expovariate(1.0 / 100) if not is_illicit else random.expovariate(1.0 / 500)
            gas_price = random.uniform(10, 100) if not is_illicit else random.uniform(100, 500)

            txs.append(
                RawTransaction(
                    tx_id=f"0x{block_height:08x}{i:04x}{random.randint(0, 0xFFFF):04x}",
                    block_height=block_height,
                    timestamp=base_time + random.randint(0, 13),
                    from_address=from_addr,
                    to_address=to_addr,
                    value=round(value, 6),
                    gas_price=round(gas_price, 2),
                    gas_used=random.randint(21000, 500000),
                    input_data=b"",
                    ingestion_ts=int(time.time()),
                )
            )
        return txs

    def stop(self):
        self._running = False
        self._producer.flush()
        self._producer.close()


class TransactionConsumer:
    """Kafka consumer for reading raw transactions."""

    def __init__(self, kafka_bootstrap_servers: str = "localhost:9092",
                 topic: str = "raw_txs", group_id: str = "kyt-consumer"):
        self._consumer = KafkaConsumer(
            topic,
            bootstrap_servers=kafka_bootstrap_servers,
            group_id=group_id,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="earliest",
            enable_auto_commit=True,
        )

    def consume(self) -> Iterator[RawTransaction]:
        for message in self._consumer:
            yield RawTransaction.from_dict(message.value)

    def close(self):
        self._consumer.close()