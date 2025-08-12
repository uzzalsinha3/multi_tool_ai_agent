import os
import sqlite3
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool, set_tracing_disabled
from tavily import TavilyClient
from pydantic import BaseModel
from typing import List

from sqlalchemy import create_engine

# Load environment variables from .env file
load_dotenv()

# Required environment variables
BASE_URL = os.getenv("BASE_URL") 
API_KEY = os.getenv("API_KEY") 
MODEL_NAME = os.getenv("MODEL_NAME") 
TAVILY_API = os.getenv("Tavily_API")

if not BASE_URL or not API_KEY or not MODEL_NAME:
    raise ValueError("Please set BASE_URL, API_KEY, and MODEL_NAME in your environment.")

# Initialize OpenAI client with async support
client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

# Disable tracing for cleaner logs/output
set_tracing_disabled(disabled=True)

# Mapping CSV files to SQLite database files
DATA_MAP = {
    "data/heart_disease.csv": "heart_disease.db",
    "data/cancer.csv": "cancer.db",
    "data/diabetes.csv": "diabetes.db",
}

def csv_to_sqlite(csv_path, db_path, table_name=None):
    """
    Convert a CSV file to a SQLite database table.
    :param csv_path: Path to the CSV file.
    :param db_path: Path to the SQLite database file.
    :param table_name: Table name to create in SQLite (defaults to CSV filename without extension).
    """
    if table_name is None:
        table_name = os.path.splitext(os.path.basename(csv_path))[0]
    engine = create_engine(f"sqlite:///{db_path}")
    df = pd.read_csv(csv_path)
    # Sanitize column names (replace spaces and hyphens with underscores)
    df.columns = [c.strip().replace(" ", "_").replace("-", "_") for c in df.columns]
    # Write DataFrame to SQLite (replace table if exists)
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    # print(f"Wrote {len(df)} rows to {db_path}:{table_name}")

# Define Pydantic models for structured web search results
class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str

class SearchResults(BaseModel):
    results: List[SearchResult]

@function_tool
def websearch(query: str) -> SearchResults:
    """
    Perform a web search using Tavily API and return structured results.
    """
    tavily_client = TavilyClient(api_key=TAVILY_API)
    response = tavily_client.search(query)
    results_list = [
        SearchResult(title=r["title"], url=r["url"], snippet=r["content"])
        for r in response.get("results", [])
    ]
    return SearchResults(results=results_list)

@function_tool
def query_heart_disease_db(query: str) -> str:
    """
    Execute SQL query on the heart disease database and return results.
    """
    conn = sqlite3.connect("heart_disease.db")
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    return str(rows)

@function_tool
def query_diabetes_disease_db(query: str) -> str:
    """
    Execute SQL query on the diabetes database and return results.
    """
    conn = sqlite3.connect("diabetes.db")
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    return str(rows)

@function_tool
def query_cancer_db(query: str) -> str:
    """
    Execute SQL query on the cancer database and return results.
    """
    conn = sqlite3.connect("cancer.db")
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    return str(rows)

# Initialize the AI assistant agent with instructions and tools
agent = Agent(
    name="Assistant",
    instructions="""
    You are a helpful assistant that can answer questions using the available tools.

    Tools:
    1. query_heart_disease_db — Use this for questions about heart disease data/statistics.
    2. query_diabetes_disease_db — Use this for diabetes database queries.
    3. query_cancer_db — Use this for cancer database queries.
    4. websearch — Use for general knowledge or current events outside of the databases.

    Rules:
    - Prefer database queries when question relates to specific datasets.
    - Use websearch for general knowledge or current events.
    - Do not invent data; only provide information from the tools.
    - Provide clear, concise answers and explain the data source.
    """,
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    tools=[query_heart_disease_db, query_diabetes_disease_db, query_cancer_db, websearch]
)

def main():
    # Convert all CSV datasets to their SQLite databases if CSV exists
    for csv_path, db_path in DATA_MAP.items():
        if not os.path.exists(csv_path):
            print(f"CSV not found: {csv_path} — skipping")
            continue
        table_name = os.path.splitext(os.path.basename(csv_path))[0]
        csv_to_sqlite(csv_path, db_path, table_name=table_name)

    # Example queries to test the agent
    queries = [
        "SELECT * FROM cancer LIMIT 2;",
        "what is cancer",
        "SELECT AVG(age) FROM heart_disease;",
        "SELECT AVG(age) FROM diabetes;"
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{'='*50}")
        print(f"Query {i}: {query}")
        print('='*50)
        result = Runner.run_sync(agent, query)
        print(f"Result: {result.final_output}")

if __name__ == "__main__":
    main()
