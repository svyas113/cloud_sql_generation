import os
import sqlite3
import json
import argparse
import glob
import psycopg2

def get_schema_graph(db_path: str, db_type: str) -> dict:
    """
    Extracts the schema graph from a database, including tables and foreign key relationships.
    """
    if db_type == "sqlite":
        return get_sqlite_schema_graph(db_path)
    elif db_type == "postgresql":
        return get_postgresql_schema_graph(db_path)
    else:
        raise ValueError("Unsupported database type")

def get_sqlite_schema_graph(db_path: str) -> dict:
    nodes, edges = [], []
    try:
        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        for table_name in tables:
            if table_name.startswith('sqlite_'):
                continue
            nodes.append({'id': table_name})
            cursor.execute(f"PRAGMA foreign_key_list(`{table_name}`);")
            for fk in cursor.fetchall():
                edges.append({
                    'source': table_name,
                    'target': fk[2],
                    'relationship': f"{table_name}.{fk[3]} = {fk[2]}.{fk[4]}"
                })
        conn.close()
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return None
    return {'nodes': nodes, 'edges': edges}

def get_postgresql_schema_graph(conn_str: str) -> dict:
    nodes, edges = [], []
    try:
        conn = psycopg2.connect(conn_str)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';
        """)
        tables = [row[0] for row in cursor.fetchall()]
        for table_name in tables:
            nodes.append({'id': table_name})
        
        cursor.execute("""
            SELECT
                tc.table_name, kcu.column_name, 
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name 
            FROM 
                information_schema.table_constraints AS tc 
                JOIN information_schema.key_column_usage AS kcu
                  ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage AS ccu
                  ON ccu.constraint_name = tc.constraint_name
            WHERE constraint_type = 'FOREIGN KEY';
        """)
        for fk in cursor.fetchall():
            edges.append({
                'source': fk[0],
                'target': fk[2],
                'relationship': f"{fk[0]}.{fk[1]} = {fk[2]}.{fk[3]}"
            })
        conn.close()
    except psycopg2.Error as e:
        print(f"PostgreSQL error: {e}")
        return None
    return {'nodes': nodes, 'edges': edges}

def process_databases(db_type: str, path: str):
    if db_type == "sqlite":
        if not os.path.isdir(path):
            print(f"Error: Root directory '{path}' not found.")
            return
        for db_id in [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]:
            db_dir = os.path.join(path, db_id)
            sqlite_file = next(glob.iglob(os.path.join(db_dir, "*.sqlite")), None)
            if sqlite_file:
                process_single_db(sqlite_file, db_type, db_dir)
    elif db_type == "postgresql":
        # Extract database name from connection string
        # Format is typically: postgresql://user:password@host:port/dbname
        try:
            db_name = path.split('/')[-1]
            # Handle case where connection string doesn't end with dbname
            if not db_name or '?' in db_name:
                # Try to extract from standard connection string format
                import re
                match = re.search(r'/([^/?]+)(\?|$)', path)
                if match:
                    db_name = match.group(1)
                else:
                    # Fallback to a default name if we can't extract it
                    db_name = "postgresql_db"
            
            print(f"Extracted database name: {db_name}")
            output_dir = os.path.join("mini_dev/Dataset", db_name)
            os.makedirs(output_dir, exist_ok=True)
            process_single_db(path, db_type, output_dir)
        except Exception as e:
            print(f"Error processing PostgreSQL database: {e}")
            import traceback
            traceback.print_exc()

def process_single_db(db_path: str, db_type: str, output_dir: str):
    print(f"\nProcessing database: {db_path}")
    schema_graph = get_schema_graph(db_path, db_type)
    if schema_graph:
        output_path = os.path.join(output_dir, "schema_graph.json")
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(schema_graph, f, indent=4)
            print(f"Successfully generated schema graph: {output_path}")
        except IOError as e:
            print(f"Error writing schema graph to {output_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate schema graphs for databases.")
    parser.add_argument('--db_type', required=True, choices=['sqlite', 'postgresql'])
    parser.add_argument('--path', required=True, help='Root directory for SQLite DBs or connection string for PostgreSQL.')
    args = parser.parse_args()
    process_databases(args.db_type, args.path)

if __name__ == "__main__":
    main()
