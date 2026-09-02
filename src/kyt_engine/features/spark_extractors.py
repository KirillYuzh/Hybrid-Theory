from __future__ import annotations
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import *
import warnings
warnings.filterwarnings("ignore")

class StatFeatureExtractor:
    def __init__(self, feature_cols=None):
        self.feature_cols = feature_cols or [f"f{i}" for i in range(1, 167)]
        self.numeric_cols = ['value', 'gas_price', 'gas_used']

    def transform(self, df: DataFrame) -> DataFrame:
        tx_df = df.select('from_address', 'to_address', 'value', 
                         'gas_price', 'gas_used', 'timestamp')
        
        stats_df = tx_df.groupBy('from_address').agg(
            count('value').alias('tx_count'),
            sum('value').alias('value_sum'),
            avg('value').alias('value_mean'),
            stddev_pop('value').alias('value_std'),
            min('value').alias('value_min'),
            max('value').alias('value_max'),
            sum('gas_price').alias('gas_price_sum'),
            avg('gas_price').alias('gas_price_mean'),
            stddev_pop('gas_price').alias('gas_price_std'),
            sum('gas_used').alias('gas_used_sum'),
            avg('gas_used').alias('gas_used_mean'),
            stddev_pop('gas_used').alias('gas_used_std')
        )
        
        stats_df = stats_df.withColumn('hour', hour('timestamp'))
        stats_df = stats_df.withColumn('day_of_week', dayofweek('timestamp'))
        
        stats_df = stats_df.withColumn('value_cv', 
                                      col('value_std') / nullif(col('value_mean'), 0))
        stats_df = stats_df.withColumn('gas_price_cv', 
                                      col('gas_price_std') / nullif(col('gas_price_mean'), 0))
        stats_df = stats_df.withColumn('gas_used_cv', 
                                      col('gas_used_std') / nullif(col('gas_used_mean'), 0))
        
        for col in stats_df.columns:
            if 'std' in col:
                stats_df = stats_df.withColumn(col, when(col.isNull(), 0).otherwise(col))
        
        result_cols = []
        for i, col in enumerate(stats_df.columns):
            if i < 166:
                result_cols.append(col)
        
        return stats_df.select(*result_cols)

class BehaviorFeatureExtractor:
    def __init__(self, windows=None):
        self.windows = windows or {"30d": 30, "60d": 60, "90d": 90}
    
    def compute_behavioral(self, features_df: DataFrame) -> DataFrame:
        result_df = features_df.groupBy('from_address').agg(
            count('timestamp').alias('tx_count_30d'),
            avg('value').alias('value_mean_30d'),
            stddev_pop('value').alias('value_std_30d'),
            avg('gas_price').alias('gas_price_mean_30d'),
            stddev_pop('gas_price').alias('gas_price_std_30d'),
            hour('timestamp').alias('hour_of_day'),
            dayofweek('timestamp').alias('day_of_week')
        )
        
        result_df = result_df.withColumn('burstiness_30d',
                                        expr('tx_count_30d / (value_mean_30d * gas_price_mean_30d + 1)'))
        result_df = result_df.withColumn('value_cv_30d',
                                        expr('value_std_30d / nullif(value_mean_30d, 0)'))
        result_df = result_df.withColumn('gas_cv_30d',
                                        expr('gas_price_std_30d / nullif(gas_price_mean_30d, 0)'))
        
        return result_df

class GraphFeatureExtractor:
    def compute_graph_features(self, edges_df: DataFrame) -> DataFrame:
        result_df = edges_df.groupBy('src').agg(
            count('*').alias('out_degree'),
            sum('amount').alias('total_amount_sent')
        ).union(
            edges_df.groupBy('dst').agg(
                count('*').alias('in_degree'),
                sum('amount').alias('total_amount_received')
            )
        )
        
        result_df = result_df.withColumn('degree_ratio',
                                        when(col('in_degree').isNull(), 0.0)
                                        .otherwise(
                                            when(col('out_degree').isNull(), 0.0)
                                            .otherwise(col('in_degree') / nullif(col('out_degree'), 1))
                                        ))
        
        return result_df.select('src', 'in_degree', 'out_degree', 'degree_ratio')

class EmbeddingGenerator:
    def train_node2vec(self, edges_df: DataFrame) -> DataFrame:
        """Generate node embeddings using Word2Vec (Node2Vec approximation).

        Returns Spark DataFrame with node and embedding columns.
        Raises ImportError if PySpark is not available.
        """
        try:
            from pyspark.ml.feature import Word2Vec
            from pyspark.sql import SparkSession
        except ImportError as e:
            raise ImportError(
                "EmbeddingGenerator requires PySpark. Install with: pip install pyspark"
            ) from e

        import pandas as pd
        from pyspark.sql.functions import monotonically_increasing_id

        unique_nodes = pd.concat([edges_df.select('src').distinct().toPandas(),
                                  edges_df.select('dst').distinct().toPandas()]).drop_duplicates()
        node_list = unique_nodes['src'].tolist() + unique_nodes['dst'].tolist()
        node_list = list(set(node_list))

        embeddings_df = pd.DataFrame({'node': node_list})

        if len(embeddings_df) == 0:
            spark = SparkSession.builder.getOrCreate()
            return spark.createDataFrame(embeddings_df)

        spark = SparkSession.builder.getOrCreate()

        edges_spark = edges_df.withColumnRenamed('src', 'src_node')
        edges_spark = edges_spark.withColumnRenamed('dst', 'dst_node')

        word2vec = Word2Vec(
            inputCol='src_node',
            outputCol='embedding',
            vectorSize=64,
            maxIter=20,
            seed=42
        )

        model = word2vec.fit(edges_spark)
        result = model.transform(edges_spark)
        result = result.select('dst_node', 'embedding')

        embeddings_spark = spark.createDataFrame(embeddings_df)
        embeddings_spark = embeddings_spark.join(result, on='dst_node', how='left')
        embeddings_spark = embeddings_spark.fillna(0.0, subset=['embedding'])
        embeddings_spark = embeddings_spark.withColumn('node', monotonically_increasing_id())

        return embeddings_spark.select('node', 'embedding')
