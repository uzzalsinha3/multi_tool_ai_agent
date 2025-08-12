# AI Assistant with Database and Web Search Integration

This project implements an AI-powered assistant that answers user queries by intelligently leveraging multiple data sources:
- Local SQLite databases generated from CSV files about **heart disease**, **cancer**, and **diabetes** datasets.
- Web search results via the **Tavily API** for general knowledge and current events.

The assistant uses the **OpenAI** API to interpret user queries and decide which tool to use to provide accurate, data-driven answers.

---

## Features

- Converts CSV datasets into SQLite databases automatically.
- Executes SQL queries on local databases for precise data retrieval.
- Uses web search to handle general or up-to-date questions.
- Implements clear tool selection logic for appropriate response sourcing.
- Uses `agents` framework with OpenAI models for conversational interaction.

---

## Requirements

###  Clone this repo

```bash
git clone https://github.com/uzzalsinha3/multi_tool_ai_agent

```

- Python 3.8 to 3.11
- Install dependencies:

```bash
pip install -r requirements.txt

###  Create `.env`

Create a `.env` file in the root

## ▶️ Run the Assistant

```bash
python multi_tool_ai_agent.py
```

---
