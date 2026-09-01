from __future__ import annotations
import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pyflink.datastream import StreamExecutionEnvironment
    from pyflink.datastream.connectors.kafka import KafkaSource, KafkaOffsetsInitializer
    from pyflink.datastream.formats.json import JsonRowDeserializationSchema
    from pyflink.table import StreamTableEnvironment
    from pyflink.common.watermark_strategy import WatermarkStrategy
    from pyflink.common.typeinformation import BasicTypeInformation

logger = logging.getLogger(__name__)


class FlinkFeatureExtractor:
    """Flink streaming job for real-time feature extraction.
    
    Reads from Kafka 'raw_txs', computes features, writes to Iceberg.
    """

    def __init__(self, kafka_bootstrap_servers: str = "localhost:9092",
                 iceberg_catalog: str = "nessie",
                 iceberg_warehouse: str = "s3://kyt-lake/warehouse"):
        self._kafka_servers = kafka_bootstrap_servers
        self._catalog = iceberg_catalog
        self._warehouse = iceberg_warehouse
        self._env: Optional[StreamExecutionEnvironment] = None
        self._tEnv: Optional[StreamTableEnvironment] = None

    def _init_env(self):
        """Lazy initialization of Flink environment."""
        if self._env is None:
            from pyflink.datastream import StreamExecutionEnvironment
            from pyflink.datastream.connectors.kafka import KafkaSource, KafkaOffsetsInitializer
            from pyflink.datastream.formats.json import JsonRowDeserializationSchema
            from pyflink.table import StreamTableEnvironment
            from pyflink.common.watermark_strategy import WatermarkStrategy
            from pyflink.common.typeinformation import BasicTypeInformation
            self._env = StreamExecutionEnvironment.get_execution_environment()
            self._tEnv = StreamTableEnvironment.create(self._env)
            # Store classes for later use
            self._KafkaSource = KafkaSource
            self._KafkaOffsetsInitializer = KafkaOffsetsInitializer
            self._JsonRowDeserializationSchema = JsonRowDeserializationSchema
            self._WatermarkStrategy = WatermarkStrategy
            self._BasicTypeInformation = BasicTypeInformation

    def configure(self):
        """Configure Flink environment."""
        self._init_env()
        self._env.set_parallelism(4)

        self._tEnv.execute_sql(f"""
            CREATE CATALOG iceberg_catalog WITH (
                'type' = 'iceberg',
                'catalog-type' = 'rest',
                'uri' = 'http://nessie:19120/api/v1',
                'warehouse' = '{self._warehouse}'
            )
        """)

        self._tEnv.execute_sql("""
            CREATE TABLE IF NOT EXISTS iceberg_catalog.kyt.features (
                tx_id STRING,
                block_height BIGINT,
                timestamp TIMESTAMP,
                from_address STRING,
                to_address STRING,
                value DOUBLE,
                gas_price DOUBLE,
                gas_used BIGINT,
                stat_feat_1 DOUBLE,
                stat_feat_2 DOUBLE,
                processing_time AS CURRENT_TIMESTAMP,
                PRIMARY KEY (tx_id) NOT ENFORCED
            ) PARTITIONED BY (hours(timestamp))
            WITH ('format-version' = '2')
        """)

        self._tEnv.execute_sql("""
            CREATE TABLE IF NOT EXISTS iceberg_catalog.kyt.raw_txs (
                tx_id STRING,
                block_height BIGINT,
                timestamp TIMESTAMP,
                from_address STRING,
                to_address STRING,
                value DOUBLE,
                gas_price DOUBLE,
                gas_used BIGINT,
                ingestion_ts TIMESTAMP,
                PRIMARY KEY (tx_id) NOT ENFORCED
            ) PARTITIONED BY (hours(timestamp))
            WITH ('format-version' = '2')
        """)

    def build_pipeline(self):
        """Build Flink streaming pipeline."""
        self._init_env()
        schema = (
            "tx_id STRING,"
            "block_height BIGINT,"
            "timestamp BIGINT,"
            "from_address STRING,"
            "to_address STRING,"
            "value DOUBLE,"
            "gas_price DOUBLE,"
            "gas_used BIGINT,"
            "input_data STRING,"
            "ingestion_ts BIGINT"
        )

        ds = self._env.from_source(
            self._KafkaSource.builder()
            .set_bootstrap_servers(self._kafka_servers)
            .set_topics("raw_txs")
            .set_starting_offsets(self._KafkaOffsetsInitializer.committed_offsets())
            .set_value_only_deserializer(self._JsonRowDeserializationSchema.builder()
                .type_info(schema)
                .build())
            .build(),
            self._WatermarkStrategy.for_monotonous_watermarks(),
            "Kafka Source"
        )

        self._tEnv.create_view("raw_stream", ds)

        self._tEnv.execute_sql("""
            INSERT INTO iceberg_catalog.kyt.raw_txs
            SELECT
                tx_id,
                block_height,
                TO_TIMESTAMP(FROM_UNIXTIME(timestamp / 1000)) as timestamp,
                from_address,
                to_address,
                value,
                gas_price,
                gas_used,
                TO_TIMESTAMP(FROM_UNIXTIME(ingestion_ts / 1000)) as ingestion_ts
            FROM raw_stream
        """)

    def run(self):
        """Submit Flink job."""
        self.configure()
        self.build_pipeline()
        self._env.execute("kyt-feature-extractor")

    def stop(self):
        self._env.cancel()