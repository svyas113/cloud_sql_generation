import json
import os
import argparse
import time
# import httpx # No longer needed for API calls
import glob
from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd
import asyncio # Needed for top-level async execution
import litellm # Import litellm
import sqlite3
import re
from datetime import datetime
import traceback
import openpyxl
from pathlib import Path
import networkx as nx
from hyde_module import HydeModule
from schema_vectorization_module import SchemaVectorizer

# Set the DeepSeek API key from environment variable or argument later
# It's often better practice to set it as an environment variable
# os.environ["DEEPSEEK_API_KEY"] = "YOUR_API_KEY"

class DeepSeekSQLGenerator:
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek/deepseek-chat", # Use litellm format
        # max_retries: int = 3, # Handled by litellm
        # retry_delay: int = 5, # Handled by litellm
        temperature: float = 0.7,
        max_tokens: int = 4096
    ):
        """
        Initialize the DeepSeek SQL Generator using litellm.

        Args:
            api_key: DeepSeek API key (will be passed to litellm)
            model: Model to use for generation (default: deepseek/deepseek-chat)
                     Use the format expected by litellm (e.g., 'deepseek/model-name').
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate in the response
        """
        self.api_key = api_key # Store the key to pass to litellm
        self.model = model
        # self.max_retries = max_retries # No longer needed directly
        # self.retry_delay = retry_delay # No longer needed directly
        self.temperature = temperature
        self.max_tokens = max_tokens
        # self.base_url = "https://api.deepseek.com/v1/chat/completions" # No longer needed

        # Optional: Configure litellm globally if needed (e.g., logging)
        # litellm.set_verbose = True
        
        # Load feedback data from first 5 feedback files
        self.feedback_data = self.load_feedback_files()
        self.hyde = HydeModule(api_key)

    def load_feedback_files(self) -> str:
        """
        Load the first 5 feedback files from the feedback folder.
        
        Returns:
            Combined content of the first 5 feedback files
        """
        feedback_dir = "feedback"
        feedback_files = []
        
        try:
            # Get all .md files from the feedback directory
            for feedback_file in glob.glob(os.path.join(feedback_dir, "*.md")):
                feedback_files.append(feedback_file)
            
            # Sort files to ensure consistent loading order
            feedback_files.sort()
            
            # Take the first 5 files (or fewer if there are less than 5)
            feedback_files = feedback_files[:5]
            
            # Load and combine the content of the files
            combined_feedback = ""
            for file_path in feedback_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    combined_feedback += f"\n\n--- Feedback from {os.path.basename(file_path)} ---\n\n{content}"
            
            print(f"Loaded {len(feedback_files)} feedback files")
            return combined_feedback
        except Exception as e:
            print(f"Warning: Could not load feedback files: {str(e)}")
            return ""
    
    async def generate_sql(self, prompt: str) -> str:
        """
        Generate SQL using the specified model via litellm.

        Args:
            prompt: The prompt to send to the model

        Returns:
            Generated SQL query or an error message
        """
        messages = [{"role": "user", "content": prompt}]

        print(f"Sending request to {self.model} via litellm...")
        # print(f"Prompt (start): {prompt[:200]}...") # Optional: print start of prompt

        try:
            # Use litellm.acompletion for asynchronous calls
            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                api_key=self.api_key, # Pass API key directly
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                # Optional: Add litellm specific parameters if needed
                # e.g., num_retries=3, timeout=120
            )

            # litellm response structure is similar to OpenAI's
            # Accessing content (check litellm docs for exact structure if needed)
            sql = response.choices[0].message.content # Access via attributes
            # Or using dictionary access: sql = response['choices'][0]['message']['content']

            print(f"Successfully received response via litellm.")
            print(f"Generated SQL (start): {sql[:100]}...") # Print first 100 chars
            return sql.strip() # Remove potential leading/trailing whitespace

        except Exception as e:
            # Catch potential exceptions from litellm (API errors, connection issues, etc.)
            print(f"Error during litellm API call: {str(e)}")
            print(f"Exception type: {type(e)}")
            import traceback
            traceback.print_exc()
            # You might want more specific error handling based on litellm exceptions
            return f"Error: Failed to generate SQL using litellm: {str(e)}"

        # Fallback error message (should ideally be caught by the exception block)
        # return "Error: Failed to generate SQL via litellm." # This line is likely unreachable now

    # --- create_prompt simplified to directly use string schema ---
    def find_deterministic_join_path(self, required_tables: List[str], schema_graph: Dict[str, Any]) -> List[str]:
        """
        Finds the most efficient join path between a list of required tables using a schema graph.

        Args:
            required_tables: A list of table names required for the query.
            schema_graph: A dictionary representing the database schema (nodes and edges).

        Returns:
            A list of formatted SQL JOIN clauses.
        """
        if not required_tables or len(required_tables) < 2:
            return []

        G = nx.Graph()
        for edge in schema_graph['edges']:
            G.add_edge(edge['source'], edge['target'], relationship=edge['relationship'])

        # Find a subgraph that connects all required tables
        subgraph_nodes = set(required_tables)
        for i in range(len(required_tables)):
            for j in range(i + 1, len(required_tables)):
                try:
                    path = nx.shortest_path(G, source=required_tables[i], target=required_tables[j])
                    for node in path:
                        subgraph_nodes.add(node)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
        
        subgraph = G.subgraph(subgraph_nodes)
        
        # Get the minimum spanning tree to ensure the most efficient path
        mst = nx.minimum_spanning_tree(subgraph)
        
        join_clauses = []
        for u, v, data in mst.edges(data=True):
            join_clauses.append(f"JOIN {v} ON {data['relationship']}")
            
        return join_clauses

    def create_prompt(self, focused_schema: str, table_contents: Dict[str, Any], question: str, join_clauses: List[str], evidence: Optional[str] = None, db_type: str = "sqlite") -> str:
        """
        Create a prompt for the LLM based on database schema and question.

        Args:
            focused_schema: Focused database schema information from HyDE.
            table_contents: Sample contents of tables.
            question: Natural language question to convert to SQL.
            join_clauses: A list of deterministic JOIN clauses to be used in the query.
            evidence: Optional evidence/context.
            db_type: Database type (sqlite, mysql, postgresql).

        Returns:
            Formatted prompt for the LLM.
        """
        # Use the focused schema string directly
        schema_str = focused_schema

        # Include sample data in the prompt
        sample_data_str = ""
        for table_name, sample_data in table_contents.items():
            if sample_data:
                sample_data_str += f"Sample data for {table_name}:\n"
                sample_data_str += sample_data
                sample_data_str += "\n"

        # Include feedback data if available
        feedback_str = ""
        if hasattr(self, 'feedback_data') and self.feedback_data:
            feedback_str = f"""
FEEDBACK FROM PREVIOUS SQL QUERY ANALYSES:
The following sections contain detailed analyses of SQL queries, including common errors and their corrections. 
Each feedback document includes:
1. Initial incorrect queries and explanations of why they were wrong
2. Corrected queries with explanations of why they work
3. Key insights about the database structure, relationships, and field usage
4. SQL best practices and common pitfalls to avoid when writing queries for this database

Use these examples to understand the database structure and write more accurate SQL queries:
{self.feedback_data}
"""

        join_str = "\n".join(join_clauses)

        prompt = f"""You are an expert SQL developer. Your task is to convert the following natural language question into a valid {db_type.upper()} SQL query.
Here is how you should approach this task:
1. Understand the question and identify the tables and columns involved.
2. Use the provided database schema and sample data to find the relevant tables and columns.
3. **You MUST use the following JOIN clauses in your query if applicable.**
4. Write a valid SQL query that answers the question.
5. Ensure the SQL query is syntactically correct and follows best practices.

IMPORTANT: Your response MUST be ONLY the SQL query, without any explanations, comments, or natural language text.
Even if you're not sure about the exact tables or columns, make your best attempt to write a SQL query based on the schema provided.
If you cannot find exact matches for table or column names, use the closest matches from the schema.
Do not include any comments in your response.
Do not start with the symbol ```
Do not include any line breaks \\n or any of the formatting or unnecessary backslashes in your response.
You only need to return the result {db_type.upper()} SQL code starting with SELECT.

HERE IS THE DATABASE SCHEMA AND SAMPLE DATA FOR EACH TABLE:
{schema_str}

DETERMINISTIC JOIN PATH:
{join_str}

{feedback_str}

"""

        prompt += f"""
QUESTION:
{question}

Please generate a valid {db_type.upper()} SQL query that answers the question. Return ONLY the SQL query without any explanations or markdown formatting.

REMEMBER: Your response must be ONLY SQL code starting with SELECT. Do not include any explanations, apologies, or statements about not being able to find tables. If you're unsure about exact table or column names, make your best guess based on the schema provided.
"""
        return prompt

    # --- load_database_schema updated to find the actual SQLite file before calling generate_schema_prompt_sqlite ---
    def load_database_schema(self, db_path: str, db_type: str) -> str:
        """
        Load database schema from database file.

        Args:
            db_path: Path to the database directory or connection string
            db_type: Type of database

        Returns:
            String containing CREATE statements for all tables
        """
        vectorizer = SchemaVectorizer(db_path, db_type)
        return "\n".join(vectorizer.get_schema_descriptions())


    # --- load_table_samples remains the same ---
    def load_table_samples(self, db_path: str, limit: int = 3) -> Dict[str, str]:
        """
        Load sample data from tables to provide context.

        Args:
            db_path: Path to the database directory
            limit: Maximum number of rows to include per table

        Returns:
            Dictionary with table samples as formatted strings
        """
        import sqlite3

        samples = {}

        # Find the SQLite database file (same logic as schema loading)
        db_file = os.path.join(db_path, f"{os.path.basename(db_path)}.sqlite")
        if not os.path.exists(db_file):
            sqlite_files = glob.glob(os.path.join(db_path, "*.sqlite"))
            if sqlite_files:
                db_file = sqlite_files[0]
            else:
                # No samples if DB not found, schema loading will warn
                return samples

        try:
            # Connect to the SQLite database
            conn = sqlite3.connect(f'file:{db_file}?mode=ro', uri=True) # Read-only mode
            conn.row_factory = sqlite3.Row  # Enable column access by name
            cursor = conn.cursor()

            # Get list of tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            for table in tables:
                table_name = table[0]

                # Skip SQLite system tables
                if table_name.startswith('sqlite_'):
                    continue

                try:
                    # Get sample data
                    # Use backticks for table names that might be keywords or contain special chars
                    cursor.execute(f"SELECT * FROM `{table_name}` LIMIT {limit};")
                    rows = cursor.fetchall()

                    if rows:
                        # Format as a simple table
                        column_names = rows[0].keys()
                        # Escape pipe characters within data if formatting for Markdown
                        escape = lambda x: str(x).replace('|', '\\|') if x is not None else "NULL"

                        table_str = "| " + " | ".join(map(escape, column_names)) + " |\n"
                        table_str += "| " + " | ".join(["---"] * len(column_names)) + " |\n"

                        for row in rows:
                            values = [escape(row[col]) for col in column_names]
                            table_str += "| " + " | ".join(values) + " |\n"

                        samples[table_name] = table_str
                except Exception as e:
                    print(f"Error getting sample data for table `{table_name}` in '{db_file}': {str(e)}")

            conn.close()

        except Exception as e:
            print(f"Error loading samples from SQLite database '{db_file}': {str(e)}")
            import traceback
            traceback.print_exc()

        return samples

def execute_sql_query(db_path: str, sql_query: str, db_type: str = "sqlite") -> Tuple[bool, Any, str, str]:
    """
    Execute a SQL query on the specified database.
    
    Args:
        db_path: Path to the database file (for SQLite) or connection string (for PostgreSQL)
        sql_query: SQL query to execute
        db_type: Database type (sqlite or postgresql)
        
    Returns:
        Tuple containing:
        - Success flag (True if query executed successfully, False otherwise)
        - Results (if successful) or None (if failed)
        - Error message (if failed) or empty string (if successful)
        - Error type (if failed) or empty string (if successful)
    """
    # Check if the SQL query is actually an error message (not a valid SQL query)
    if sql_query.strip().lower().startswith("i cannot generate"):
        return False, None, "Response contains an error message, not a valid SQL query", "InvalidQueryError"
    
    conn = None
    try:
        if db_type == "sqlite":
            conn = sqlite3.connect(db_path)
        elif db_type == "postgresql":
            # Assumes db_path is a psycopg2 connection string
            # e.g., "dbname=test user=postgres password=secret"
            import psycopg2
            conn = psycopg2.connect(db_path)
        else:
            return False, None, "Unsupported database type", "ConfigurationError"

        cursor = conn.cursor()
        cursor.execute(sql_query)  # This will raise an exception if the SQL is invalid
        
        if sql_query.strip().upper().startswith("SELECT"):
            results = cursor.fetchall()
            column_names = [description[0] for description in cursor.description]
            df = pd.DataFrame(results, columns=column_names)
            return True, df, "", ""
        else:
            conn.commit()
            affected_rows = cursor.rowcount
            return True, f"Query executed successfully. Affected rows: {affected_rows}", "", ""
            
    except Exception as e:
        error_message = str(e)
        error_type = type(e).__name__
        return False, None, error_message, error_type
    finally:
        if conn:
            conn.close()

def extract_column_names_from_schema(db_schema: str) -> Dict[str, List[str]]:
    """
    Extract all table and column names from the database schema.
    
    Args:
        db_schema: Database schema information as a string with CREATE statements
        
    Returns:
        Dictionary mapping table names to lists of column names
    """
    table_columns = {}
    
    # Regular expression to find CREATE TABLE statements
    create_table_pattern = r"CREATE TABLE\s+(\w+)\s*\((.*?)\)\s*(?:/\*.*?\*/\s*)?(?=CREATE TABLE|\Z)"
    
    # Find all CREATE TABLE statements in the schema
    for match in re.finditer(create_table_pattern, db_schema, re.DOTALL | re.IGNORECASE):
        table_name = match.group(1)
        columns_definition = match.group(2)
        
        # Extract column names from the columns definition
        column_pattern = r"[`\"]([^`\"]+)[`\"]|\b(\w+)\b(?=\s+(?:TEXT|INTEGER|REAL|DATE|double|varchar))"
        columns_matches = re.findall(column_pattern, columns_definition)
        
        # Process the matches to get single column names
        columns = []
        for col_tuple in columns_matches:
            # Take the first non-empty value from the tuple
            col_name = next((name for name in col_tuple if name), None)
            if col_name and col_name.lower() not in ['primary', 'foreign', 'key', 'references']:
                columns.append(col_name)
        
        table_columns[table_name] = columns
    
    return table_columns

def create_error_correction_prompt(db_schema: str, question: str, sql_query: str, 
                                  error_message: str, db_type: str = "sqlite") -> str:
    """
    Create a prompt for the LLM to fix a SQL query that resulted in an error.
    
    Args:
        db_schema: Database schema information as a string with CREATE statements
        question: Original natural language question
        sql_query: SQL query that caused the error
        error_message: Error message from the database
        db_type: Database type (sqlite, mysql, postgresql)
        
    Returns:
        Formatted prompt for the LLM
    """
    # Use the schema string directly
    schema_str = db_schema
    
    # Check if the error is about a missing column
    no_such_column_match = re.search(r"no such column:?\s+(\S+)", error_message, re.IGNORECASE)
    
    if no_such_column_match:
        # Extract the column name from the error message
        full_column_ref = no_such_column_match.group(1)
        
        # Handle table alias if present (e.g., "s.SchoolType" -> "SchoolType")
        if '.' in full_column_ref:
            table_alias, missing_column = full_column_ref.split('.', 1)
        else:
            missing_column = full_column_ref
        
        # Extract all table and column names from the schema
        table_columns = extract_column_names_from_schema(db_schema)
        
        # Create a list of all columns in the format "table_name.column_name"
        all_columns = []
        for table_name, columns in table_columns.items():
            all_columns.extend(columns)
        
        prompt = f"""You are an expert SQL developer. Your task is to fix a {db_type.upper()} SQL query that resulted in an error.

DATABASE SCHEMA AND SAMPLE DATA FOR EACH TABLE:
{schema_str}

ORIGINAL NATURAL LANGAUGE QUESTION:
{question}

ORIGINAL SQL QUERY:
{sql_query}

ERROR MESSAGE:
{error_message}

Please fix the SQL query to use the correct column name. Return ONLY the corrected SQL query without any explanations or markdown formatting.
"""
    else:
        # Use the regular prompt for other types of errors
        prompt = f"""You are an expert SQL developer. Your task is to fix a {db_type.upper()} SQL query that resulted in an error.

DATABASE SCHEMA AND SAMPLE DATA FOR EACH TABLE:
{schema_str}

ORIGINAL QUESTION:
{question}

ORIGINAL SQL QUERY:
{sql_query}

ERROR MESSAGE:
{error_message}

Please fix the SQL query to resolve the error. Return ONLY the corrected SQL query without any explanations or markdown formatting.
"""
    print(prompt)
    return prompt

async def sql_feedback_loop(
    generator: DeepSeekSQLGenerator,
    db_path: str,
    db_schema: str,
    question: str,
    difficulty: str,
    db_id: str,
    db_type: str = "sqlite",
    max_attempts: int = 5
) -> Dict[str, Any]:
    """
    Implement a feedback loop for SQL query generation and execution with HyDE-based regeneration.
    
    Args:
        generator: DeepSeekSQLGenerator instance
        db_path: Path to the SQLite database file
        db_schema: Database schema information
        question: Natural language question
        difficulty: Question difficulty level
        db_id: Database identifier
        max_attempts: Maximum number of attempts to fix the query
        
    Returns:
        Dictionary with results of the feedback loop
    """
    # Create initial prompt and generate SQL
    prompt = generator.create_prompt(db_schema, {}, question, [])
    sql_query = await generator.generate_sql(prompt)
    
    feedback_results = {
        "question": question,
        "question_difficulty": difficulty,
        "database": db_id,
        "attempts": []
    }
    
    # Try executing the query with up to max_attempts
    for attempt in range(1, max_attempts + 1):
        print(f"Attempt {attempt}/{max_attempts} - Executing SQL query...")
        
        # Record this attempt
        current_attempt = {
            "attempt_number": attempt,
            "sql_query": sql_query,
            "success": False,
            "error_message": "",
            "error_type": ""
        }
        
        # Execute the query
        success, results, error_message, error_type = execute_sql_query(db_path, sql_query, db_type)
        
        # Update attempt record
        current_attempt["success"] = success
        current_attempt["error_message"] = error_message
        current_attempt["error_type"] = error_type
        
        # Add to feedback results
        feedback_results["attempts"].append(current_attempt)
        
        if success:
            print(f"SQL query executed successfully on attempt {attempt}!")
            feedback_results["final_status"] = "Success"
            feedback_results["iteration_at_pass"] = attempt
            feedback_results["final_sql"] = sql_query
            feedback_results["results"] = results
            break
        else:
            print(f"SQL query failed on attempt {attempt}: {error_message}")
            
            # If we've reached max attempts, stop
            if attempt >= max_attempts:
                print(f"Maximum attempts ({max_attempts}) reached. Query could not be fixed.")
                feedback_results["final_status"] = "Failed"
                feedback_results["iteration_at_pass"] = "Didn't Pass"
                feedback_results["final_sql"] = sql_query
                break
            
            # For subsequent attempts, use HyDE to regenerate the focused schema
            print(f"Using HyDE to regenerate focused schema for attempt {attempt+1}...")
            
            # Get a new focused schema using HyDE with the enhanced prompt that includes error info
            new_focused_schema = await generator.hyde.get_focused_schema_with_error(
                question, db_id, sql_query, error_message
            )
            
            # Log the HyDE regeneration process
            log_dir = "Feedback loop Errors/prompt_logs"
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"{log_dir}/hyde_regeneration_{db_id}_{attempt}_{timestamp}.txt"
            
            with open(log_filename, "w", encoding="utf-8") as log_file:
                log_file.write(f"Database: {db_id}\n")
                log_file.write(f"Question: {question}\n")
                log_file.write(f"Attempt: {attempt}\n")
                log_file.write(f"Error Message: {error_message}\n")
                log_file.write(f"Original SQL: {sql_query}\n\n")
                log_file.write(f"New Focused Schema:\n{new_focused_schema}\n")
            
            print(f"HyDE regeneration logged to {log_filename}")
            
            # Generate a new SQL query with the updated schema
            prompt = generator.create_prompt(new_focused_schema, {}, question, [])
            print(f"Generating new SQL query for attempt {attempt+1} with HyDE-focused schema...")
            sql_query = await generator.generate_sql(prompt)
    
    return feedback_results

def save_feedback_to_excel(feedback_results: List[Dict[str, Any]], output_file: str):
    """
    Save feedback results to an Excel file.
    
    Args:
        feedback_results: List of feedback result dictionaries
        output_file: Path to save the Excel file
    """
    # Create a DataFrame to store the results
    data = []
    
    for result in feedback_results:
        question = result["question"]
        difficulty = result["question_difficulty"]
        database = result.get("database", "Unknown")
        final_status = result.get("final_status", "Unknown")
        iteration_at_pass = result.get("iteration_at_pass", "Didn't Pass")
        
        # Process each attempt
        for attempt in result["attempts"]:
            data.append({
                "Question": question,
                "Question Difficulty": difficulty,
                "Database": database,
                "Attempt Number": attempt["attempt_number"],
                "SQL Query": attempt["sql_query"],
                "Success": attempt["success"],
                "Error Message": attempt["error_message"],
                "Error Type": attempt["error_type"],
                "Final Status": final_status,
                "Iteration @ Pass": iteration_at_pass if attempt["attempt_number"] == iteration_at_pass else ""
            })
    
    # Create DataFrame for the main feedback sheet
    df = pd.DataFrame(data)
    
    # Create analytics data
    analytics_data = generate_analytics(feedback_results)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save to Excel with multiple sheets
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Feedback Results', index=False)
        analytics_data.to_excel(writer, sheet_name='Analytics', index=False)
    
    print(f"Feedback results saved to {output_file}")

def generate_analytics(feedback_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Generate analytics from feedback results.
    
    Args:
        feedback_results: List of feedback result dictionaries
        
    Returns:
        DataFrame with analytics data
    """
    total_queries = len(feedback_results)
    successful_queries = sum(1 for result in feedback_results if result.get("final_status") == "Success")
    unsuccessful_queries = total_queries - successful_queries
    
    # Count queries by number of attempts needed
    success_by_attempt = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    
    for result in feedback_results:
        if result.get("final_status") == "Success":
            iteration = result.get("iteration_at_pass")
            if isinstance(iteration, int) and 1 <= iteration <= 5:
                success_by_attempt[iteration] += 1
    
    # Create analytics DataFrame
    analytics = [
        {"Metric": "Total Queries Processed", "Value": total_queries},
        {"Metric": "Successful Queries", "Value": successful_queries},
        {"Metric": "Unsuccessful Queries (after 5 attempts)", "Value": unsuccessful_queries},
        {"Metric": "Queries Successful in First Attempt", "Value": success_by_attempt[1]},
        {"Metric": "Queries Successful in Second Attempt", "Value": success_by_attempt[2]},
        {"Metric": "Queries Successful in Third Attempt", "Value": success_by_attempt[3]},
        {"Metric": "Queries Successful in Fourth Attempt", "Value": success_by_attempt[4]},
        {"Metric": "Queries Successful in Fifth Attempt", "Value": success_by_attempt[5]}
    ]
    
    return pd.DataFrame(analytics)

def extract_tables_from_question(question: str, schema_tables: List[str]) -> List[str]:
    """
    A simple heuristic to extract table names from the question.
    This can be replaced with a more sophisticated method like HyDE.
    """
    required_tables = []
    for table in schema_tables:
        if table.lower() in question.lower():
            required_tables.append(table)
    return required_tables

# --- process_dataset remains largely the same, just uses the updated generator ---
async def process_dataset(
    api_key: str,
    input_file: str,
    output_file: str,
    db_dir: Optional[str] = None,
    model: str = "deepseek/deepseek-chat", # Default updated
    db_type: str = "sqlite",
    db_connection_string: Optional[str] = None,
    limit: Optional[int] = None,
    feedback_loop: bool = True,
    excel_output: Optional[str] = None
):
    """
    Process the entire dataset and generate SQL queries using litellm.

    Args:
        api_key: DeepSeek API key
        input_file: Path to input JSON file with questions and database info
        output_file: Path to output file to save results
        db_dir: Directory containing database files (e.g., BIRD/dev_databases)
        model: Model to use for generation (litellm format, e.g., deepseek/deepseek-chat)
        db_type: Database type (sqlite, mysql, postgresql)
        limit: Optional limit on number of questions to process
        feedback_loop: Whether to use the feedback loop for SQL execution and correction
        excel_output: Path to save feedback results to Excel file
    """
    # Load the dataset
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_file}")
        return

    # Initialize the generator using the potentially updated model name
    generator = DeepSeekSQLGenerator(api_key=api_key, model=model)
    results = []  # Initialize as empty list to store results
    feedback_results = []

    # Cache for database schemas, samples, and graphs
    db_schemas = {}
    db_samples = {}
    db_files = {}  # Cache for database file paths
    db_graphs = {} # Cache for schema graphs

    # Process each question
    questions_to_process = dataset[:limit] if limit else dataset
    total_questions = len(questions_to_process)

    for i, item in enumerate(questions_to_process):
        start_time = time.time()
        print("-" * 50)
        print(f"Processing question {i+1}/{total_questions}")

        # --- Database ID extraction (remains the same) ---
        db_id = item.get('db_id', None)
        if not db_id:
             # Adapt based on your input file structure (e.g., for BIRD dev.json)
             # If db_id is directly in the item, use it.
             # Otherwise, you might need to look elsewhere or construct it.
             # Example for BIRD structure might involve looking at a path or related fields.
             # Using a placeholder if not found:
             db_id = f"unknown_db_in_item_{i+1}"
             print(f"Warning: 'db_id' not found directly in item {i+1}. Using placeholder: {db_id}")
             # You might need more sophisticated logic depending on dev.json format

        # Extract question
        question = item.get('question', item.get('nl', '')) # Handle potential key differences
        if not question:
             print(f"Warning: Could not find question ('question' or 'nl') in item {i+1}. Skipping.")
             continue # Skip if no question

        # Extract difficulty (if available)
        difficulty = item.get('difficulty', 'unknown')

        # Extract evidence (not used in prompt currently, but good to have)
        evidence = item.get('evidence', '')

        print(f"Database ID: {db_id}")
        print(f"Question: {question}")
        print(f"Difficulty: {difficulty}")

        # --- Schema and Sample Loading ---
        if db_id not in db_schemas:
            print(f"Loading schema and samples for {db_id}...")
            
            # Handle PostgreSQL and SQLite differently
            if db_type == 'postgresql':
                if not db_connection_string:
                    print(f"Error: PostgreSQL connection string is required but not provided")
                    continue
                
                # For PostgreSQL, use the connection string directly
                db_path_for_vectorizer = db_connection_string
                
                # Create a directory for schema graph if needed
                os.makedirs("mini_dev/Dataset", exist_ok=True)
                pg_db_dir = os.path.join("mini_dev/Dataset", db_id)
                os.makedirs(pg_db_dir, exist_ok=True)
                
                # Load samples directly from PostgreSQL
                db_samples[db_id] = {}  # Initialize empty, will be populated if needed
                
                # Check for schema graph in the PostgreSQL directory
                schema_graph_path = os.path.join(pg_db_dir, "schema_graph.json")
                
                # Set db_file to connection string for PostgreSQL
                db_files[db_id] = db_connection_string
            else:
                # For SQLite, use the traditional path-based approach
                if not db_dir:
                    print(f"Error: Database directory is required for SQLite but not provided")
                    continue
                    
                current_db_path = os.path.join(db_dir, db_id)
                db_path_for_vectorizer = os.path.join(current_db_path, f"{db_id}.sqlite")
                
                # Load samples from SQLite
                db_samples[db_id] = generator.load_table_samples(current_db_path)
                
                # Check for schema graph in the SQLite directory
                schema_graph_path = os.path.join(current_db_path, "schema_graph.json")
                
                # Find the SQLite database file
                db_file = os.path.join(current_db_path, f"{os.path.basename(current_db_path)}.sqlite")
                if not os.path.exists(db_file):
                    sqlite_files = glob.glob(os.path.join(current_db_path, "*.sqlite"))
                    if sqlite_files:
                        db_file = sqlite_files[0]
                        print(f"Found database file: {db_file}")
                    else:
                        print(f"Warning: No SQLite database file found in {current_db_path}")
                        db_file = None
                
                db_files[db_id] = db_file
            
            # Vectorize schema (works for both PostgreSQL and SQLite)
            vectorizer = SchemaVectorizer(db_path_for_vectorizer, db_type)
            vectorizer.vectorize_and_store_schema()
            
            # Get focused schema using HyDE (works for both PostgreSQL and SQLite)
            db_schemas[db_id] = await generator.hyde.get_focused_schema(question, db_id)
            
            # Load the schema graph if it exists
            if os.path.exists(schema_graph_path):
                with open(schema_graph_path, 'r') as f:
                    db_graphs[db_id] = json.load(f)
            else:
                db_graphs[db_id] = None
                print(f"Warning: schema_graph.json not found for {db_id}")

            if not db_schemas[db_id]:
                 print(f"Warning: Failed to load schema for {db_id}. SQL generation might be inaccurate.")

        focused_schema = db_schemas[db_id]
        table_samples = db_samples[db_id]
        db_file = db_files[db_id]
        schema_graph = db_graphs.get(db_id)

        if schema_graph:
            # Extract tables and find join path
            schema_tables = [node['id'] for node in schema_graph['nodes']]
            required_tables = extract_tables_from_question(question, schema_tables)
            join_clauses = generator.find_deterministic_join_path(required_tables, schema_graph)
        else:
            join_clauses = []

        if feedback_loop and db_file:
            # Use the enhanced feedback loop for SQL generation and execution
            print(f"Using feedback loop for SQL generation and execution...")
            db_path_for_execution = db_connection_string if db_type == 'postgresql' else db_file
            
            # Call the sql_feedback_loop function with the focused schema
            feedback_result = await sql_feedback_loop(
                generator=generator,
                db_path=db_path_for_execution,
                db_schema=focused_schema,
                question=question,
                difficulty=difficulty,
                db_id=db_id,
                db_type=db_type,
                max_attempts=5  # Set to 5 as per requirements
            )
            
            # Add the feedback result to our collection
            feedback_results.append(feedback_result)
            
            # Use the final SQL from the feedback loop
            sql = feedback_result.get("final_sql", "")

        else:
            # Standard SQL generation without feedback loop
            print(f"Generating SQL without feedback loop...")
            prompt = generator.create_prompt(focused_schema, table_samples, question, join_clauses, None, db_type)
            sql = await generator.generate_sql(prompt)

        # --- Result Formatting ---
        try:
            result = {
                "db_id": db_id,
                "question": question,
                "sql": sql if sql else "Error: Failed to generate SQL", # Store the generated SQL or error message
                "model": model,
                "db_type": db_type,
                "difficulty": difficulty,
                # "evidence": evidence # Optional: include evidence if needed later
            }
            results.append(result)
        except Exception as e:
            print(f"Error formatting result: {e}")
            import traceback
            traceback.print_exc()

        end_time = time.time()
        print(f"Time taken for question {i+1}: {end_time - start_time:.2f} seconds")

        # --- Intermediate Saving ---
        if (i + 1) % 10 == 0 or (i + 1) == total_questions:
            print(f"\nSaving intermediate results ({i+1}/{total_questions})...")
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2)
                print(f"Results saved to {output_file}")

                # Also save evaluation format
                eval_output = output_file.replace('.json', '_eval.json')
                format_for_evaluation(results, eval_output)
                print(f"Evaluation format saved to {eval_output}")
                
                # Save feedback results to Excel if specified
                if feedback_loop and excel_output:
                    save_feedback_to_excel(feedback_results, excel_output)
                    print(f"Feedback results saved to {excel_output}")
            except Exception as e:
                print(f"Error saving intermediate results: {e}")
                traceback.print_exc()
            print("-" * 50)

    # Final save (might be redundant if last item triggered intermediate save)
    print("\nProcessing complete. Saving final results...")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"Final results saved to {output_file}")

        eval_output = output_file.replace('.json', '_eval.json')
        format_for_evaluation(results, eval_output)
        print(f"Final evaluation format saved to {eval_output}")
        
        # Save feedback results to Excel if specified
        if feedback_loop and excel_output:
            save_feedback_to_excel(feedback_results, excel_output)
            print(f"Final feedback results saved to {excel_output}")
    except Exception as e:
        print(f"Error saving final results: {e}")
        traceback.print_exc()

    # Ensure we return valid results even if there were errors
    if not results:
        results = []
    if not feedback_results:
        feedback_results = []
    
    return results, feedback_results

# --- format_for_evaluation remains the same ---
def format_for_evaluation(results: List[Dict[str, Any]], output_file: str):
    """
    Format results in the format required by the BIRD evaluation script.

    Args:
        results: List of result dictionaries
        output_file: Path to save formatted results
    """
    formatted_results = {}
    result_format = "\t----- bird -----\t" # Standard BIRD separator
    for idx, item in enumerate(results):
        # Ensure SQL is a string, even if it's an error message
        sql_output = str(item.get('sql', 'Error: SQL not generated'))
        db_id_output = item.get('db_id', 'unknown_db')
        # Format: { "index": "SQL<separator>db_id" }
        formatted_results[str(idx)] = f"{sql_output}{result_format}{db_id_output}"

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_results, f, indent=4) # Use indent 4 for readability
    except Exception as e:
        print(f"Error saving evaluation format file '{output_file}': {e}")


# --- main function updated for litellm model name ---
async def main():
    parser = argparse.ArgumentParser(description='Generate SQL queries using DeepSeek API via litellm')
    parser.add_argument('--api_key', required=True, help='DeepSeek API key')
    parser.add_argument('--input_file', required=True, help='Input JSON file with questions (e.g., BIRD/dev.json)')
    parser.add_argument('--output_file', required=True, help='Output JSON file to save detailed results')
    parser.add_argument('--db_dir', help='Directory containing database folders (e.g., BIRD/dev_databases)')
    # Updated default and help text for model
    parser.add_argument('--model', default='deepseek/deepseek-chat',
                        help="Model to use via litellm (e.g., 'deepseek/deepseek-chat', 'deepseek/deepseek-coder')")
    parser.add_argument('--db_connection_string', help='Connection string for PostgreSQL.')
    parser.add_argument('--db_type', default='sqlite', choices=['sqlite', 'mysql', 'postgresql'],
                        help='Database type (affects prompt instructions)')
    parser.add_argument('--limit', type=int, help='Limit number of questions to process (for testing)')
    parser.add_argument('--feedback_loop', action='store_true', default=True,
                        help='Enable feedback loop for SQL execution and correction')
    parser.add_argument('--no_feedback_loop', action='store_false', dest='feedback_loop',
                        help='Disable feedback loop for SQL execution and correction')
    parser.add_argument('--excel_output', default='Feedback loop Errors/feedback_results(Descriptive feedback loop).xlsx',
                        help='Path to save feedback results to Excel file')

    args = parser.parse_args()

    # You could also load API key from environment here as an alternative
    # api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
    # if not api_key:
    #     print("Error: DeepSeek API key is required. Set via --api_key or DEEPSEEK_API_KEY environment variable.")
    #     return

    print("Starting SQL generation process...")
    print(f"Using model: {args.model}")
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Database directory: {args.db_dir}")
    print(f"Feedback loop: {'Enabled' if args.feedback_loop else 'Disabled'}")
    if args.feedback_loop:
        print(f"Excel output: {args.excel_output}")
    if args.limit:
        print(f"Processing limit: {args.limit} questions")

    await process_dataset(
        api_key=args.api_key,
        input_file=args.input_file,
        output_file=args.output_file,
        db_dir=args.db_dir,
        model=args.model,
        db_type=args.db_type,
        db_connection_string=args.db_connection_string,
        limit=args.limit,
        feedback_loop=args.feedback_loop,
        excel_output=args.excel_output if args.feedback_loop else None
    )
    print("SQL generation process finished.")

if __name__ == "__main__":
    # Ensure you have litellm installed: pip install litellm
    # You might also need specific DB drivers if not using sqlite, e.g., pip install psycopg2-binary mysql-connector-python
    asyncio.run(main())
