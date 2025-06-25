import asyncio

import pandas as pd
import plotly.io as pio
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import set_global_handler
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from sqlalchemy import MetaData, create_engine

from components.query_runner import QueryRunner
from components.sql_storage import SQLStorage
from components.stock_data_source import StockDataSource
from utils.time_utils import get_current_date

load_dotenv()
pio.renderers.default = "browser"
pd.options.plotting.backend = "plotly"


set_global_handler("langfuse")


def show_plotly_chart(chart_json: str):
    """
    Displays a Plotly chart from a JSON string, should be called
    after running a query that returns a Plotly JSON response.
    """
    chart = pio.from_json(chart_json)

    st.plotly_chart(chart, use_container_width=True)


async def main():
    st.title("Hỏi đáp chứng khoán")
    engine = create_engine("duckdb:///:memory:")
    metadata_obj = MetaData()

    db_storage = SQLStorage(
        engine=engine,
        metadata_obj=metadata_obj,
    )
    data_source = StockDataSource(db_storage=db_storage)
    data_source.pull_symbols()
    query_runner = QueryRunner(engine=engine, llm=OpenAI(model="gpt-4o"))

    all_tools = data_source.all_tools + [
        query_runner.run_query,
        get_current_date,
        show_plotly_chart,
    ]

    workflow = FunctionAgent(
        tools=all_tools,
        llm=OpenAI(
            model="gpt-4o",
        ),
        system_prompt=(
            "You are a helpful assistant that can answer questions about stock listings and quotes. "
            "You can pull stock listings and quotes from the Vnstock API, run SQL queries on Pandas DataFrames, "
            "and provide information about the data stored in the database."
            "You can also clear the storage and get all table descriptions if user want to create a conversation with you."
            "Note that data is not available at start, so you need to pull listings and quotes first before querying them."
            "You can also visualize the data using Plotly charts but only for the final result of the query, not intermediate steps."
            "If some function return empy dictionary, it means that no data is available for the query or we dont have to query database for this query."
            "If user greets or says something not related to stock listings or quotes, you can just say hello back or ask how you can help, only focus to work with stock listings and quotes, "
            "and do not try to answer questions that are not related to stock listings or quotes."
        ),
        verbose=True,
    )

    prompt = st.chat_input("Say something and/or attach an image")

    if prompt:
        # Show user input
        st.write(f"**User:** {prompt}")

        with st.spinner("Thinking..."):
            response = await workflow.run(prompt)

            st.write(str(response))


if __name__ == "__main__":
    asyncio.run(main())
