"""PostgreSQL database for the assumption-monitor project."""

from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
class DatabaseError(Exception):
    """Raised when a database operation or query is invalid."""
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
    func,
    select,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://project_user:project123@localhost:5432/project_db",
)

engine = create_engine(DATABASE_URL)


# ============================================================
# DATABASE TABLES
# ============================================================

class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String, primary_key=True)
    file_name: Mapped[str] = mapped_column(String, nullable=False)
    n_sentences: Mapped[int] = mapped_column(Integer, nullable=False)
    loaded_at_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )


class Sentence(Base):
    __tablename__ = "sentences"

    sentence_id: Mapped[str] = mapped_column(String, primary_key=True)

    doc_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("documents.doc_id"),
        nullable=False,
    )

    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    pages: Mapped[str | None] = mapped_column(Text)
    section: Mapped[str | None] = mapped_column(Text)

    sentence: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    n_words: Mapped[int] = mapped_column(Integer, nullable=False)
    n_occurrences: Mapped[int] = mapped_column(Integer, nullable=False)

    percents: Mapped[str | None] = mapped_column(Text)
    money: Mapped[str | None] = mapped_column(Text)

    extraction_status: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )


class Series(Base):
    __tablename__ = "series"

    series_id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
    )

    title: Mapped[str | None] = mapped_column(String)
    units: Mapped[str | None] = mapped_column(String)
    frequency: Mapped[str | None] = mapped_column(String)
    topic: Mapped[str | None] = mapped_column(String)


class Observation(Base):
    __tablename__ = "observations"

    series_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("series.series_id"),
        primary_key=True,
    )

    observation_date: Mapped[date] = mapped_column(
        Date,
        primary_key=True,
    )

    available_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    value: Mapped[float | None] = mapped_column(Float)
    n_daily_obs: Mapped[int | None] = mapped_column(Integer)

    is_missing: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
    )

    is_filled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
    )

    is_outlier: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
    )

    robust_z: Mapped[float | None] = mapped_column(Float)
    source_sha256: Mapped[str | None] = mapped_column(String)


# ============================================================
# BASIC HELPERS
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


def init_db():
    """Create the four PostgreSQL tables if they do not exist."""

    Base.metadata.create_all(engine)

    print("Database tables created successfully.")


# ============================================================
# DOCUMENT LOADING
# ============================================================

def load_document(result, file_name):
    """
    Store one document and its sentences.

    Existing sentences for the same document are replaced.
    Everything happens inside one transaction.
    """

    sentences = result.sentences

    if sentences.empty:
        raise ValueError("No sentences to store.")

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

        # Remove old sentences for this document.
        session.query(Sentence).filter(
            Sentence.doc_id == doc_id
        ).delete(
            synchronize_session=False
        )

        session.add_all(rows)

        session.commit()

    return len(rows)


def load_documents(file_paths):
    """Read and load multiple PDF/DOCX documents."""

    from assumption_monitor import reader

    total = 0

    for file_path in file_paths:

        path = Path(file_path)

        print(f"Loading document: {path}")

        result = reader.read_document(path)

        count = load_document(
            result,
            path.name,
        )

        total += count

        print(
            f"  Loaded {count} sentences from {path.name}"
        )

        for warning in result.warnings:
            print(f"  WARNING: {warning}")

    print(
        f"Finished loading documents. "
        f"Total sentences: {total}"
    )

    return total


# ============================================================
# SERIES / OBSERVATION LOADING
# ============================================================

def load_observations(
    series_id,
    table,
    entry,
    source_sha256="",
):
    """
    Store cleaned weekly observations.

    Existing observations for the same series are replaced.
    """

    rows = []

    for row in table.itertuples():

        observation_date = pd.to_datetime(
            row.observation_date
        ).date()

        available_at = pd.to_datetime(
            row.available_at
        )

        if available_at.tzinfo is None:
            available_at = available_at.tz_localize(
                "UTC"
            )

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

        indicator = session.get(
            Series,
            series_id,
        )

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

        session.query(Observation).filter(
            Observation.series_id == series_id
        ).delete(
            synchronize_session=False
        )

        session.add_all(rows)

        session.commit()

    return len(rows)


def load_series(series_ids):
    """
    Clean existing registered series and load them
    into PostgreSQL.
    """

    from assumption_monitor import series
    from assumption_monitor import sources

    total = 0

    registry = sources.load_registry()

    for series_id in series_ids:

        print(f"Loading series: {series_id}")

        entry = sources.get_entry(
            registry,
            series_id,
        )

        table = series.clean_series(
            series_id
        )

        source_sha256 = ""

        try:
            source_sha256 = sources.series_source_sha256(
                series_id
            )
        except AttributeError:
            # Older sources.py versions may not have
            # this helper. Loading still works.
            pass

        count = load_observations(
            series_id=series_id,
            table=table,
            entry=entry,
            source_sha256=source_sha256,
        )

        total += count

        print(
            f"  Loaded {count} observations"
        )

    print(
        f"Finished loading series. "
        f"Total observations: {total}"
    )

    return total


# ============================================================
# QUERIES
# ============================================================

def observations_as_of(series_id, as_of):
    """
    Return observations that were available
    on or before the supplied date/time.
    """

    if isinstance(as_of, str):
        as_of = pd.to_datetime(as_of)

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
            .order_by(
                Observation.observation_date
            )
        )

        rows = session.execute(
            statement
        ).all()

    return pd.DataFrame(
        rows,
        columns=[
            "observation_date",
            "available_at",
            "value",
            "is_filled",
            "is_outlier",
        ],
    )


def table_counts():
    """Return row counts for all four main tables."""

    with Session(engine) as session:

        return {
            "documents": session.query(
                Document
            ).count(),

            "sentences": session.query(
                Sentence
            ).count(),

            "series": session.query(
                Series
            ).count(),

            "observations": session.query(
                Observation
            ).count(),
        }


def run_query(sql):
    """
    Run a read-only SQL query and return a DataFrame.
    """

    if not sql.lstrip().lower().startswith(("select", "with")):
        raise DatabaseError("Only SELECT or WITH queries are allowed.")
    ...

    sql_clean = sql.strip().lower()

    if not (
        sql_clean.startswith("select")
        or sql_clean.startswith("with")
    ):
        raise ValueError(
            "Only SELECT/WITH read-only queries are allowed."
        )

    with engine.connect() as connection:

        return pd.read_sql(
            text(sql),
            connection,
        )


# ============================================================
# COMMAND LINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Assumption Monitor PostgreSQL database"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # init
    subparsers.add_parser(
        "init",
        help="Create database tables",
    )

    # counts
    subparsers.add_parser(
        "counts",
        help="Show row counts",
    )

    # load-docs
    docs_parser = subparsers.add_parser(
        "load-docs",
        help="Load PDF/DOCX documents",
    )

    docs_parser.add_argument(
        "files",
        nargs="+",
    )

    # load-series
    series_parser = subparsers.add_parser(
        "load-series",
        help="Load cleaned indicator series",
    )

    series_parser.add_argument(
        "series_ids",
        nargs="+",
    )

    # query
    query_parser = subparsers.add_parser(
        "query",
        help="Run a read-only SQL query",
    )

    query_parser.add_argument(
        "sql",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # INIT
    # --------------------------------------------------------

    if args.command == "init":

        init_db()
        return 0

    # --------------------------------------------------------
    # COUNTS
    # --------------------------------------------------------

    if args.command == "counts":

        init_db()

        counts = table_counts()

        for table_name, count in counts.items():

            print(
                f"{table_name}: {count}"
            )

        return 0

    # --------------------------------------------------------
    # LOAD DOCUMENTS
    # --------------------------------------------------------

    if args.command == "load-docs":

        init_db()

        load_documents(
            args.files
        )

        return 0

    # --------------------------------------------------------
    # LOAD SERIES
    # --------------------------------------------------------

    if args.command == "load-series":

        init_db()

        load_series(
            args.series_ids
        )

        return 0

    # --------------------------------------------------------
    # QUERY
    # --------------------------------------------------------

    if args.command == "query":

        result = run_query(
            args.sql
        )

        if result.empty:
            print("(no rows)")
        else:
            print(
                result.to_string(
                    index=False
                )
            )

        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
