# Cloud SQL Generation Pipeline

A robust natural language to SQL query generation system using HyDE (Hypothetical Document Embeddings) and large language models. This system is designed to work with PostgreSQL databases in Google Cloud.

## Architecture

The system consists of four main components:

1. **Schema Vectorization**: Converts database schema into vector representations for efficient searching.
2. **HyDE (Hypothetical Document Embeddings)**: Identifies the most relevant parts of the schema for a given question.
3. **Schema Graph Generation**: Creates a graph representation of the database schema to determine efficient JOIN paths.
4. **SQL Generation**: Integrates all components to generate accurate SQL queries using LLMs.

## Features

- **PostgreSQL Support**: Optimized for Google Cloud PostgreSQL databases.
- **Schema Vectorization**: Efficiently indexes and searches database schema using vector embeddings.
- **HyDE-Powered Relevance**: Uses hypothetical document embeddings to find the most relevant schema parts.
- **Intelligent JOIN Path Detection**: Automatically determines the most efficient JOIN paths between tables.
- **Error Correction**: Implements a feedback loop to detect and fix SQL errors.
- **Batch Processing**: Supports processing large numbers of queries in batches.

## Prerequisites

- Python 3.11+
- PostgreSQL database (Google Cloud SQL instance)
- DeepSeek API key or other LLM API key compatible with LiteLLM

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/svyas113/cloud_sql_generation.git
   cd cloud_sql_generation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   # Create a .env file with your API key and database connection string
   echo "API_KEY=your_api_key_here" > .env
   echo "DB_CONNECTION_STRING=postgresql://username:password@34.93.119.106:5432/oncomx_import" >> .env
   ```

## Usage

### 1. Prepare Input Data

Convert your natural language questions to the required format:

```bash
python extract_nl_sql.py
python jsonl_convertion.py
python jsonl_to_gold.py
```

### 2. Generate Schema Vectors

```bash
python schema_vectorization_module.py --db_type postgresql --db_path "postgresql://username:password@34.93.119.106:5432/oncomx_import"
```

### 3. Generate Schema Graphs

```bash
python create_schema_graphs.py --db_type postgresql --path "postgresql://username:password@34.93.119.106:5432/oncomx_import"
```

### 4. Generate SQL Queries

```bash
python run_sql_generation.py --input_file gold.json --output_file results.json
```

### 5. Process in Batch Mode

For large numbers of queries, use the batch processing script:

```bash
python batch_process.py --input_file combined_question_sql_pairs.json --output_file results.json --batch_size 50
```

## Configuration

The main configuration parameters can be set in the `.env` file or passed as command-line arguments:

- `API_KEY`: Your DeepSeek API key or other LLM API key
- `DB_CONNECTION_STRING`: PostgreSQL connection string
- `MODEL`: LLM model to use (default: "deepseek/deepseek-chat")
- `BATCH_SIZE`: Number of queries to process in each batch

## File Structure

- `schema_vectorization_module.py`: Converts database schema to vector representations
- `hyde_module.py`: Implements the HyDE technique for focused schema generation
- `create_schema_graphs.py`: Generates graph representations of database schemas
- `deepseek_sql_generator_finalV5.py`: Main SQL generation script
- `extract_nl_sql.py`: Extracts natural language questions and SQL pairs from JSON files
- `jsonl_convertion.py`: Converts JSON to JSONL format
- `jsonl_to_gold.py`: Converts JSONL to the gold format for evaluation
- `run_sql_generation.py`: Script to run the SQL generation pipeline
- `batch_process.py`: Script for batch processing of queries

## Google Cloud Deployment

This system is designed to be deployed on Google Cloud Platform. For optimal performance and cost-effectiveness, we recommend using a Spot VM instance for batch processing:

```bash
gcloud compute instances create sql-generation-vm \
  --project=your-project-id \
  --zone=asia-south1-a \
  --machine-type=e2-standard-4 \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP
```

## License

[MIT License](LICENSE)

## Acknowledgments

- This project uses the DeepSeek API for language model capabilities
- Vector embeddings are generated using the Sentence Transformers library
- Vector storage is handled by LanceDB
