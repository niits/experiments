from typing import Dict, Optional

import pandas as pd
from sqlalchemy import Column, MetaData, String, Table, select, text
from sqlalchemy.engine import Engine


class SQLStorage:
    def __init__(self, engine: Engine, metadata_obj: MetaData):
        self.engine = engine
        self.metadata_obj = metadata_obj

        with self.engine.connect() as conn:
            conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS table_descriptions ("
                    "table_name TEXT PRIMARY KEY, "
                    "description TEXT)"
                )
            )
            conn.commit()

    def save_dataframe(
        self, df: pd.DataFrame, table_name: str, description: Optional[str] = None
    ):
        df.to_sql(table_name, self.engine, if_exists="fail", index=False)
        if description:
            self._save_table_description(table_name, description)

    def get_dataframe(self, table_name: str) -> pd.DataFrame:
        return pd.read_sql_query(f"SELECT * FROM {table_name}", self.engine)

    def remove_table(self, table_name: str):
        with self.engine.connect() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))

            conn.execute(
                text("DELETE FROM table_descriptions WHERE table_name = :table_name"),
                {"table_name": table_name},
            )
            conn.commit()

    def _save_table_description(self, table_name: str, description: str):
        with self.engine.connect() as conn:
            # Create the table description if it does not exist
            conn.execute(
                text(
                    "INSERT OR REPLACE INTO table_descriptions (table_name, description) "
                    "VALUES (:table_name, :description)"
                ),
                {"table_name": table_name, "description": description},
            )
            # Commit the changes to the database
            conn.commit()

    def get_table_description(self, table_name: str) -> Optional[str]:
        with self.engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT description FROM table_descriptions WHERE table_name = :table_name"
                ),
                {"table_name": table_name},
            ).scalar()
            # If the table description does not exist, return None
            if result is None:
                return None

            schema = conn.execute(
                text(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name=:table_name"  # noqa: E501
                ),
                {"table_name": table_name},
            ).scalar()

            if schema is not None or isinstance(schema, str):
                result = (
                    f"{result}\nSchema: {schema.replace('CREATE TABLE', '').strip()}"
                )

            return result

    def get_all_table_descriptions(self) -> Dict[str, str]:
        with self.engine.connect() as conn:

            result = conn.execute(
                text("SELECT table_name, description FROM table_descriptions")
            ).fetchall()

            def get_schema(table_name: str) -> str:
                schema = conn.execute(
                    text(
                        "SELECT sql FROM sqlite_master WHERE type='table' AND name=:table_name"  # noqa: E501
                    ),
                    {"table_name": table_name},
                ).scalar()
                return (
                    f"\nSchema: {schema.replace('CREATE TABLE', '').strip()}"
                    if schema is not None and isinstance(schema, str)
                    else ""
                )

            return {
                row.table_name: f"{row.description}{get_schema(row.table_name)}"
                for row in result
                if row.table_name != "table_descriptions"
            }

    def check_table_exists(self, table_name: str) -> bool:
        with self.engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=:table_name"  # noqa: E501
                ),
                {"table_name": table_name},
            ).fetchone()
            return result is not None
