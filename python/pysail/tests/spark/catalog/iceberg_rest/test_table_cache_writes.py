"""Writes through a Sail server with the catalog table cache enabled.

Server `cached` caches loaded tables for an hour. Server `other` has no cache and plays another
client of the same catalog: it changes each table after `cached` has read (and cached) it.
Every write through `cached` must then work against the current table, never the cached one,
and the result is checked through `other`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pysail.testing.spark.session import spark_connect_server, spark_session_factory

if TYPE_CHECKING:
    from collections.abc import Generator

    from pyspark.sql import SparkSession


NAMESPACE = "iceberg_table_cache_writes"

ROW_LEVEL_MODES = {
    "copy-on-write": "",
    "merge-on-read": """
        TBLPROPERTIES (
          'format-version' = '2',
          'write.delete.mode' = 'merge-on-read',
          'write.update.mode' = 'merge-on-read',
          'write.merge.mode' = 'merge-on-read'
        )
    """,
}


def _server_envs(iceberg_rest_endpoint: str, seaweedfs_host_endpoint: str, cache: str) -> dict[str, str]:
    catalog_config = f'[{{name="sail", type="iceberg-rest", uri="{iceberg_rest_endpoint}"{cache}}}]'
    return {
        "SAIL_CATALOG__LIST": catalog_config,
        "AWS_ACCESS_KEY_ID": "admin",
        "AWS_SECRET_ACCESS_KEY": "password",
        "AWS_REGION": "us-east-1",
        "AWS_ENDPOINT": seaweedfs_host_endpoint,
        "AWS_VIRTUAL_HOSTED_STYLE_REQUEST": "false",
        "AWS_ALLOW_HTTP": "true",
    }


@pytest.fixture(scope="module")
def cached(
    iceberg_rest_endpoint: str,
    seaweedfs_host_endpoint: str,
) -> Generator[SparkSession, None, None]:
    cache = ', table_cache_type="global", table_cache_ttl_secs=3600'
    envs = _server_envs(iceberg_rest_endpoint, seaweedfs_host_endpoint, cache)
    with spark_connect_server(envs=envs) as server, spark_session_factory(server.remote) as sessions:
        yield sessions.create()


@pytest.fixture(scope="module")
def other(
    iceberg_rest_endpoint: str,
    seaweedfs_host_endpoint: str,
) -> Generator[SparkSession, None, None]:
    envs = _server_envs(iceberg_rest_endpoint, seaweedfs_host_endpoint, "")
    with spark_connect_server(envs=envs) as server, spark_session_factory(server.remote) as sessions:
        session = sessions.create()
        session.sql(f"CREATE DATABASE IF NOT EXISTS {NAMESPACE}")
        yield session
        session.sql(f"DROP DATABASE IF EXISTS {NAMESPACE} CASCADE")


def _rows(spark: SparkSession, table: str, columns: str = "id, name") -> list[tuple]:
    return [tuple(row) for row in spark.sql(f"SELECT {columns} FROM {table} ORDER BY id").collect()]  # noqa: S608


def _cached_then_changed_elsewhere(
    cached: SparkSession,
    other: SparkSession,
    name: str,
    properties: str = "",
) -> str:
    """Creates a table holding row 1, has `cached` read (and cache) it, then has `other` add row 2."""
    table = f"{NAMESPACE}.{name}"
    other.sql(f"DROP TABLE IF EXISTS {table}")
    other.sql(f"CREATE TABLE {table} (id INT, name STRING) USING iceberg {properties}")
    other.sql(f"INSERT INTO {table} VALUES (1, 'first')")  # noqa: S608
    assert _rows(cached, table) == [(1, "first")]
    other.sql(f"INSERT INTO {table} VALUES (2, 'second')")  # noqa: S608
    return table


def test_reads_are_cached(cached: SparkSession, other: SparkSession) -> None:
    # Without this, the tests below could pass only because nothing is cached.
    table = _cached_then_changed_elsewhere(cached, other, "reads_are_cached")
    assert _rows(cached, table) == [(1, "first")]
    assert _rows(other, table) == [(1, "first"), (2, "second")]


def test_insert_values(cached: SparkSession, other: SparkSession) -> None:
    table = _cached_then_changed_elsewhere(cached, other, "insert_values")
    cached.sql(f"INSERT INTO {table} VALUES (3, 'third')")  # noqa: S608
    assert _rows(other, table) == [(1, "first"), (2, "second"), (3, "third")]


def test_insert_select_from_itself(cached: SparkSession, other: SparkSession) -> None:
    table = _cached_then_changed_elsewhere(cached, other, "insert_select_self")
    cached.sql(f"INSERT INTO {table} SELECT id + 10, name FROM {table}")  # noqa: S608
    assert _rows(other, table) == [(1, "first"), (2, "second"), (11, "first"), (12, "second")]


def test_insert_overwrite_from_itself(cached: SparkSession, other: SparkSession) -> None:
    table = _cached_then_changed_elsewhere(cached, other, "insert_overwrite_self")
    cached.sql(f"INSERT OVERWRITE TABLE {table} SELECT id, 'kept' FROM {table}")  # noqa: S608
    assert _rows(other, table) == [(1, "kept"), (2, "kept")]


def test_dataframe_insert_into(cached: SparkSession, other: SparkSession) -> None:
    table = _cached_then_changed_elsewhere(cached, other, "df_insert_into")
    cached.createDataFrame([(3, "third")], schema="id INT, name STRING").write.insertInto(table)
    assert _rows(other, table) == [(1, "first"), (2, "second"), (3, "third")]


def test_dataframe_append(cached: SparkSession, other: SparkSession) -> None:
    table = _cached_then_changed_elsewhere(cached, other, "df_append")
    df = cached.createDataFrame([(3, "third")], schema="id INT, name STRING")
    df.write.format("iceberg").mode("append").saveAsTable(table)
    assert _rows(other, table) == [(1, "first"), (2, "second"), (3, "third")]


def test_dataframe_overwrite(cached: SparkSession, other: SparkSession) -> None:
    table = _cached_then_changed_elsewhere(cached, other, "df_overwrite")
    df = cached.createDataFrame([(3, "third")], schema="id INT, name STRING")
    df.write.format("iceberg").mode("overwrite").saveAsTable(table)
    assert _rows(other, table) == [(3, "third")]


def test_create_table_as_select(cached: SparkSession, other: SparkSession) -> None:
    table = _cached_then_changed_elsewhere(cached, other, "ctas_source")
    target = f"{NAMESPACE}.ctas_target"
    other.sql(f"DROP TABLE IF EXISTS {target}")
    cached.sql(f"CREATE TABLE {target} USING iceberg AS SELECT * FROM {table}")  # noqa: S608
    assert _rows(other, target) == [(1, "first"), (2, "second")]


@pytest.mark.parametrize("mode", list(ROW_LEVEL_MODES))
def test_update(cached: SparkSession, other: SparkSession, mode: str) -> None:
    table = _cached_then_changed_elsewhere(cached, other, f"update_{mode.replace('-', '_')}", ROW_LEVEL_MODES[mode])
    cached.sql(f"UPDATE {table} SET name = 'updated'")
    assert _rows(other, table) == [(1, "updated"), (2, "updated")]


@pytest.mark.parametrize("mode", list(ROW_LEVEL_MODES))
def test_delete(cached: SparkSession, other: SparkSession, mode: str) -> None:
    table = _cached_then_changed_elsewhere(cached, other, f"delete_{mode.replace('-', '_')}", ROW_LEVEL_MODES[mode])
    cached.sql(f"DELETE FROM {table} WHERE id = 1")  # noqa: S608
    assert _rows(other, table) == [(2, "second")]


@pytest.mark.parametrize("mode", list(ROW_LEVEL_MODES))
def test_merge(cached: SparkSession, other: SparkSession, mode: str) -> None:
    table = _cached_then_changed_elsewhere(cached, other, f"merge_{mode.replace('-', '_')}", ROW_LEVEL_MODES[mode])
    cached.createDataFrame([(2, "merged"), (3, "inserted")], schema="id INT, name STRING").createOrReplaceTempView(
        "merge_source"
    )
    cached.sql(
        f"""
        MERGE INTO {table} AS t
        USING merge_source AS s
        ON t.id = s.id
        WHEN MATCHED THEN UPDATE SET name = s.name
        WHEN NOT MATCHED THEN INSERT *
        """
    )
    assert _rows(other, table) == [(1, "first"), (2, "merged"), (3, "inserted")]


def test_insert_after_schema_changed_elsewhere(cached: SparkSession, other: SparkSession) -> None:
    table = _cached_then_changed_elsewhere(cached, other, "schema_changed")
    evolved = other.createDataFrame([(4, "fourth", 40)], schema="id INT, name STRING, extra INT")
    evolved.write.format("iceberg").mode("append").option("mergeSchema", "true").saveAsTable(table)
    cached.sql(f"INSERT INTO {table} VALUES (3, 'third', 30)")  # noqa: S608
    assert _rows(other, table, "id, name, extra") == [
        (1, "first", None),
        (2, "second", None),
        (3, "third", 30),
        (4, "fourth", 40),
    ]
