from functools import lru_cache

from llama_index.core import SQLDatabase
from llama_index.core.llms import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import NLSQLTableQueryEngine
from sqlalchemy.engine import Engine

# noqa: E501
DEFAULT_RESPONSE_SYNTHESIS_PROMPT_TMPL = (
    "Given an input question, and query result. generate Plotly JSON could be used to render this result.\n"
    "You must choose the appropriate visualization type, ensure that the visualization has a clear title, and that the axes are labeled correctly. You must also ensure that the visualization is clear and easy to understand.\n"
    "Title and axes labels should be same language as the query.\n"
    "Only return the validPlotly JSON, do not return any other text. Don't wrap the JSON in any other format, such as a code block or a string.\n"
    "If the query result is empty, return an empty JSON object: `{}`.\n"
    "If the query result is not suitable for visualization, return an empty JSON object: `{}`.\n"
    "You will be fired if you return invalid JSON.\n"
    "Query: {query_str}\n"
    "SQL: {sql_query}\n"
    "SQL Response: {context_str}\n"
    "Response: "
)
DEFAULT_RESPONSE_SYNTHESIS_PROMPT = PromptTemplate(
    DEFAULT_RESPONSE_SYNTHESIS_PROMPT_TMPL,
)


class QueryRunner:
    """
    A class to run SQL queries on a database using LLMs.
    It uses SQLAlchemy to connect to the database and LlamaIndex
    to process natural language queries into SQL queries.
    The results are returned as strings
    """

    def __init__(self, engine: Engine, llm: LLM):
        self.engine = engine
        self.llm = llm
        self.get_sql_query_engine = lru_cache(maxsize=None)(self._get_sql_query_engine)

    def _get_sql_query_engine(self, tables: tuple) -> NLSQLTableQueryEngine:
        sql_database = SQLDatabase(self.engine, include_tables=list(tables))
        return NLSQLTableQueryEngine(
            sql_database=sql_database,
            llm=self.llm,
            response_synthesis_prompt=DEFAULT_RESPONSE_SYNTHESIS_PROMPT,
        )

    def run_query(self, tables: list[str], query: str) -> str:
        """
        Runs a SQL query on the specified table and returns the result
        as a string.
        """
        query_engine = self.get_sql_query_engine(tuple(sorted(tables)))

        response = query_engine.query(query)

        return str(response) if response is not None else "No results found."
