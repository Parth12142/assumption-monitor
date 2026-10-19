import os

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from assumption_monitor import database


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://project_user:project123@localhost:5432/project_db",
)


@pytest.fixture
def engine():
    return create_engine(DATABASE_URL)


def test_database_connection(engine):
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1")).scalar()

    assert result == 1


def test_required_tables_exist(engine):
    expected = {
        "documents",
        "sentences",
        "series",
        "observations",
    }

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                """
            )
        )

        actual = {row[0] for row in rows}

    assert expected.issubset(actual)


def test_documents_are_loaded(engine):
    with engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM documents")
        ).scalar()

    assert count >= 2


def test_sentences_are_loaded(engine):
    with engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM sentences")
        ).scalar()

    assert count >= 67


def test_series_are_loaded(engine):
    with engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM series")
        ).scalar()

    assert count >= 2


def test_observations_are_loaded(engine):
    with engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM observations")
        ).scalar()

    assert count >= 4934


def test_document_sentence_relationship(engine):
    with engine.connect() as conn:
        count = conn.execute(
            text(
                """
                SELECT COUNT(*)
                FROM sentences s
                LEFT JOIN documents d
                    ON s.doc_id = d.doc_id
                WHERE d.doc_id IS NULL
                """
            )
        ).scalar()

    assert count == 0


def test_observation_series_relationship(engine):
    with engine.connect() as conn:
        count = conn.execute(
            text(
                """
                SELECT COUNT(*)
                FROM observations o
                LEFT JOIN series s
                    ON o.series_id = s.series_id
                WHERE s.series_id IS NULL
                """
            )
        ).scalar()

    assert count == 0


def test_no_duplicate_document_ids(engine):
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT doc_id, COUNT(*)
                FROM documents
                GROUP BY doc_id
                HAVING COUNT(*) > 1
                """
            )
        ).fetchall()

    assert rows == []


def test_no_duplicate_series_ids(engine):
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT series_id, COUNT(*)
                FROM series
                GROUP BY series_id
                HAVING COUNT(*) > 1
                """
            )
        ).fetchall()

    assert rows == []


def test_no_duplicate_observations(engine):
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT series_id, observation_date, COUNT(*)
                FROM observations
                GROUP BY series_id, observation_date
                HAVING COUNT(*) > 1
                """
            )
        ).fetchall()

    assert rows == []


def test_as_of_query(engine):
    with engine.connect() as conn:
        total = conn.execute(
            text(
                """
                SELECT COUNT(*)
                FROM observations
                WHERE series_id = 'DCOILWTICO'
                """
            )
        ).scalar()

        known = conn.execute(
            text(
                """
                SELECT COUNT(*)
                FROM observations
                WHERE series_id = 'DCOILWTICO'
                AND available_at <= '2024-01-10'
                """
            )
        ).scalar()

    assert known <= total


def test_read_only_query():
    result = database.run_query(
        "SELECT COUNT(*) AS n FROM sentences"
    )

    assert "n" in result.columns
    assert result["n"].iloc[0] >= 67


def test_delete_query_is_rejected():
    with pytest.raises(Exception):
        database.run_query("DELETE FROM sentences")


def test_update_query_is_rejected():
    with pytest.raises(Exception):
        database.run_query(
            "UPDATE sentences SET sentence = 'test'"
        )


def test_insert_query_is_rejected():
    with pytest.raises(Exception):
        database.run_query(
            "INSERT INTO series (series_id) VALUES ('TEST')"
        )
