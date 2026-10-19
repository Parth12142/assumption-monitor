"""PostgreSQL storage for sentences and weekly observations (implementation plan, section 12).

The database address is read from the DATABASE_URL environment variable.
It is never written in the code.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    select,
    text,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from assumption_monitor import reader, series, sources


class DatabaseError(Exception):
    """A problem with the database or a query."""


# ============================================================
# CONNECTION
# ============================================================

def _load_env_file():
    """Read KEY=VALUE lines from the project's .env file (variables already set win)."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_file()
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise DatabaseError(
        "DATABASE_URL is not set. Put it in the .env file, "
        "then run: set -a; source .env; set +a"
    )

engine = create_engine(DATABASE_URL, pool_pre_ping=True)


# ============================================================
# TABLES
# ============================================================

class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String, primary_key=True)
    file_name: Mapped[str] = mapped_column(String, nullable=False)
    n_sentences: Mapped[int] = mapped_column(Integer, nullable=False)
    loaded_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class Sentence(Base):
    __tablename__ = "sentences"

    sentence_id: Mapped[str] = mapped_column(String, primary_key=True)
    doc_id: Mapped[str] = mapped_column(String, ForeignKey("documents.doc_id"), nullable=False)

    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    pages: Mapped[str | None] = mapped_column(Text)
    section: Mapped[str | None] = mapped_column(Text)

    sentence: Mapped[str] = mapped_column(Text, nullable=False)

    n_words: Mapped[int] = mapped_column(Integer, nullable=False)
    n_occurrences: Mapped[int] = mapped_column(Integer, nullable=False)

    percents: Mapped[str | None] = mapped_column(Text)
    money: Mapped[str | None] = mapped_column(Text)

    extraction_status: Mapped[str] = mapped_column(String, nullable=False)


class Series(Base):
    __tablename__ = "series"

    series_id: Mapped[str] = mapped_column(String, primary_key=True)

    title: Mapped[str | None] = mapped_column(String)
    units: Mapped[str | None] = mapped_column(String)
    frequency: Mapped[str | None] = mapped_column(String)
    topic: Mapped[str | None] = mapped_column(String)


class Observation(Base):
    __tablename__ = "observations"

    series_id: Mapped[str] = mapped_column(String, ForeignKey("series.series_id"), primary_key=True)
    observation_date: Mapped[date] = mapped_column(Date, primary_key=True)

    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    value: Mapped[float | None] = mapped_column(Float)
    n_daily_obs: Mapped[int | None] = mapped_column(Integer)

    is_missing: Mapped[bool] = mapped_column(Boolean, nullable=False)
    is_filled: Mapped[bool] = mapped_column(Boolean, nullable=False)
    is_outlier: Mapped[bool] = mapped_column(Boolean, nullable=False)

    robust_z: Mapped[float | None] = mapped_column(Float)
    source_sha256: Mapped[str | None] = mapped_column(String)


# ============================================================
# SMALL HELPERS
# ============================================================

def _utc_now():
    return datetime.now(timezone.utc)


def _int_or_none(value):
    if pd.isna(value):
        return None
    return int(value)


def _float_or_none(value):
    if pd.isna(value):
        return None
    return float(value)


def _source_sha(series_id):
    """Checksum of the newest stored raw file for this series (from the Day 5 download log)."""
    log = sources.read_log()
    rows = log[log["series_id"] == series_id]
    if rows.empty:
        return ""
    return rows.sort_values("retrieved_at_utc").iloc[-1]["sha256"]


def init_db():
    """Create the four tables if they do not exist yet. Safe to run many times."""
    Base.metadata.create_all(engine)


# ============================================================
# STORE SENTENCES
# ============================================================

def load_document(result, file_name):
    """Store one document and its sentences.

    Old sentences of the same document are replaced.
    Everything happens in one transaction: all rows are saved, or none.
    """
    sentences = result.sentences

    if sentences.empty:
        raise DatabaseError("No sentences to store.")

    doc_id = str(sentences["doc_id"].iloc[0])

    rows = []
    for row in sentences.itertuples():
        percents = row.percents
        money = row.money

        if not isinstance(percents, str):
            percents = json.dumps(list(percents))
        if not isinstance(money, str):
            money = json.dumps(list(money))

        rows.append(
            Sentence(
                sentence_id=str(row.sentence_id),
                doc_id=str(row.doc_id),
                page_start=_int_or_none(row.page_start),
                page_end=_int_or_none(row.page_end),
                pages=row.pages,
                section=row.section,
                sentence=row.sentence,
                n_words=int(row.n_words),
                n_occurrences=int(row.n_occurrences),
                percents=percents,
                money=money,
                extraction_status=row.extraction_status,
            )
        )

    with Session(engine) as session:
        document = session.get(Document, doc_id)

        if document is None:
            document = Document(
                doc_id=doc_id,
                file_name=str(file_name),
                n_sentences=len(rows),
                loaded_at_utc=_utc_now(),
            )
            session.add(document)
        else:
            document.file_name = str(file_name)
            document.n_sentences = len(rows)
            document.loaded_at_utc = _utc_now()

        # Remove the old sentences of this document, then add the new ones.
        session.query(Sentence).filter(Sentence.doc_id == doc_id).delete(synchronize_session=False)
        session.add_all(rows)

        session.commit()

    return len(rows)


def load_documents(file_paths):
    """Read PDF or Word files and store them."""
    total = 0

    for file_path in file_paths:
        path = Path(file_path)
        print(f"Loading document: {path}")

        result = reader.read_document(path)
        count = load_document(result, path.name)
        total += count

        print(f"  Loaded {count} sentences from {path.name}")
        for warning in result.warnings:
            print(f"  WARNING: {warning}")

    print(f"Finished loading documents. Total sentences: {total}")
    return total


# ============================================================
# STORE WEEKLY OBSERVATIONS
# ============================================================

def load_observations(series_id, table, entry, source_sha256=""):
    """Store a cleaned weekly table. Old rows of the same series are replaced."""
    rows = []

    for row in table.itertuples():
        observation_date = pd.to_datetime(row.observation_date).date()

        available_at = pd.to_datetime(row.available_at)
        if available_at.tzinfo is None:
            available_at = available_at.tz_localize("UTC")
        available_at = available_at.to_pydatetime()

        rows.append(
            Observation(
                series_id=series_id,
                observation_date=observation_date,
                available_at=available_at,
                value=_float_or_none(row.value),
                n_daily_obs=_int_or_none(row.n_daily_obs),
                is_missing=bool(row.is_missing),
                is_filled=bool(row.is_filled),
                is_outlier=bool(row.is_outlier),
                robust_z=_float_or_none(row.robust_z),
                source_sha256=source_sha256,
            )
        )

    with Session(engine) as session:
        indicator = session.get(Series, series_id)

        if indicator is None:
            indicator = Series(
                series_id=series_id,
                title=entry.get("title"),
                units=entry.get("units"),
                frequency="weekly (W-FRI)",
                topic=entry.get("topic"),
            )
            session.add(indicator)
        else:
            indicator.title = entry.get("title")
            indicator.units = entry.get("units")
            indicator.frequency = "weekly (W-FRI)"
            indicator.topic = entry.get("topic")

        session.query(Observation).filter(Observation.series_id == series_id).delete(
            synchronize_session=False
        )
        session.add_all(rows)

        session.commit()

    return len(rows)


def load_series(series_ids):
    """Clean the stored raw series and load the weekly tables."""
    total = 0
    registry = sources.load_registry()

    for series_id in series_ids:
        print(f"Loading series: {series_id}")

        entry = sources.get_entry(registry, series_id)
        table = series.clean_series(series_id, registry=registry)

        count = load_observations(
            series_id=series_id,
            table=table,
            entry=entry,
            source_sha256=_source_sha(series_id),
        )
        total += count

        print(f"  Loaded {count} observations")

    print(f"Finished loading series. Total observations: {total}")
    return total


# ============================================================
# QUESTIONS
# ============================================================

def observations_as_of(series_id, as_of):
    """Only the weeks that were already published on the `as_of` date."""
    as_of = pd.Timestamp(as_of)
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")
    as_of = as_of.to_pydatetime()

    with Session(engine) as session:
        statement = (
            select(
                Observation.observation_date,
                Observation.available_at,
                Observation.value,
                Observation.is_filled,
                Observation.is_outlier,
            )
            .where(
                Observation.series_id == series_id,
                Observation.available_at <= as_of,
            )
            .order_by(Observation.observation_date)
        )
        rows = session.execute(statement).all()

    return pd.DataFrame(
        rows,
        columns=["observation_date", "available_at", "value", "is_filled", "is_outlier"],
    )


def table_counts():
    """Number of rows in each of the four tables."""
    with Session(engine) as session:
        return {
            "documents": session.query(Document).count(),
            "sentences": session.query(Sentence).count(),
            "series": session.query(Series).count(),
            "observations": session.query(Observation).count(),
        }


def run_query(sql):
    """Run a read-only SQL query and return a DataFrame.

    Two guards: the text must start with SELECT or WITH, and the whole
    transaction is set to READ ONLY, so PostgreSQL itself refuses any change
    (even a DELETE hidden inside a WITH query).
    """
    if not sql.lstrip().lower().startswith(("select", "with")):
        raise DatabaseError("Only SELECT or WITH queries are allowed.")

    try:
        with engine.connect() as connection:
            connection.execute(text("SET TRANSACTION READ ONLY"))
            return pd.read_sql(text(sql), connection)
    except (SQLAlchemyError, pd.errors.DatabaseError) as exc:
        raise DatabaseError(f"Query failed: {exc}") from exc


# ============================================================
# COMMAND LINE
# ============================================================

def main(argv=None):
    parser = argparse.ArgumentParser(description="Assumption Monitor PostgreSQL database")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="Create the tables")
    subparsers.add_parser("counts", help="Show row counts")

    docs_parser = subparsers.add_parser("load-docs", help="Load PDF or DOCX documents")
    docs_parser.add_argument("files", nargs="+")

    series_parser = subparsers.add_parser("load-series", help="Load cleaned weekly series")
    series_parser.add_argument("series_ids", nargs="+")

    query_parser = subparsers.add_parser("query", help="Run a read-only SQL query")
    query_parser.add_argument("sql")

    args = parser.parse_args(argv)

    try:
        if args.command == "init":
            init_db()
            print("Tables are ready.")

        elif args.command == "counts":
            init_db()
            for table_name, count in table_counts().items():
                print(f"{table_name}: {count}")

        elif args.command == "load-docs":
            init_db()
            load_documents(args.files)

        elif args.command == "load-series":
            init_db()
            load_series(args.series_ids)

        elif args.command == "query":
            result = run_query(args.sql)
            if result.empty:
                print("(no rows)")
            else:
                pd.set_option("display.width", 200)
                pd.set_option("display.max_colwidth", 80)
                print(result.to_string(index=False))

    except (
        DatabaseError,
        reader.ExtractionError,
        reader.UnsupportedFileError,
        series.SeriesError,
        sources.SourceError,
        SQLAlchemyError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
