import pandas as pd
from vnstock import Listing, Quote


class StockDataSource:
    """
    A data source for stock listings and quotes.
    This class interacts with the Vnstock API to retrieve stock listings and quotes.
    It provides methods to pull symbols, industries, and quotes,
    and save them to a database using the provided db_storage instance."""

    def __init__(self, db_storage):
        self.db_storage = db_storage
        self._listing = Listing()

    def _describe_dataframe(
        self, df: pd.DataFrame, additional_metadata: dict[str, str]
    ) -> str:
        """
        Returns a DataFrame with the description of the specified table.
        """
        return "\n".join(
            [
                f"This table contains: {df.columns.tolist()}",
                f"Its statistics:\n {df.describe().to_string()}",
                f"Queried with these parameters: {additional_metadata}",
            ]
        )

    def pull_symbols(self) -> str:
        """
        Pull all stock listings from the Vnstock API and save them to the database.
        Returns a string indicating the table name.
        """
        if self.db_storage.check_table_exists("symbols"):
            return "symbols"

        df = self._listing.symbols_by_industries()

        industry_code_cols = [
            column for column in df.columns if "icb_code" in column.lower()
        ]

        industry_name_cols = [
            column for column in df.columns if "icb_name" in column.lower()
        ]

        df["industry_codes"] = df[industry_code_cols].apply(
            lambda row: row[industry_code_cols].dropna().tolist(), axis=1
        )
        df["industry_names"] = df[industry_name_cols].apply(
            lambda row: row[industry_name_cols].dropna().tolist(), axis=1
        )
        df = df.drop(columns=industry_code_cols + industry_name_cols)

        self.db_storage.save_dataframe(
            df=df,
            table_name="symbols",
            description=self._describe_dataframe(
                df,
                additional_metadata=dict(
                    source="Vnstock API",
                ),
            ),
        )

        return "symbols"

    def pull_industries(self) -> str:
        """Pulls all industries from the Vnstock API and saves them to the database.
        Returns a string indicating the table name.
        """
        if self.db_storage.check_table_exists("industries"):
            return "industries"

        df = self._listing.industries_icb()

        self.db_storage.save_dataframe(
            df=df,
            table_name="industries",
            description=self._describe_dataframe(
                df,
                additional_metadata=dict(
                    source="Vnstock API",
                ),
            ),
        )

        return "industries"

    def pull_quotes(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1D",
    ) -> str:
        """
        Pulls stock quotes for a given symbol and date range from the Vnstock API
        and saves them to the database.
        Start and end dates should be in the format YYYY-MM-DD.
        Interval can be one of ["1m", "5m", "15m", "30m", "1H", "1D", "1W", "1M"]
        Returns a string indicating the table name.
        """
        table_name = f"{symbol}_quotes_{start_date}__{end_date}__{interval}".replace(
            "-", "_"
        )
        if self.db_storage.check_table_exists(table_name):
            return table_name

        quote = Quote(symbol=symbol, source="VCI")

        df = quote.history(
            start=start_date,
            end=end_date,
            interval=interval,
        )

        self.db_storage.save_dataframe(
            df=df,
            table_name=table_name,
            description=self._describe_dataframe(
                df,
                additional_metadata=dict(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval,
                ),
            ),
        )
        return table_name

    def get_all_table_descriptions(self) -> str:
        """
        Returns all table descriptions from the database.
        """
        text = ""
        for table, metadata in self.db_storage.get_all_table_descriptions().items():
            text += f"Table: {table}\nDescription:\n{metadata}\n\n"
        return text

    @property
    def all_tools(self):
        return [
            self.pull_symbols,
            self.pull_quotes,
            self.get_all_table_descriptions,
        ]
