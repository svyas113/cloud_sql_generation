import os
import sqlite3
import lancedb
from sentence_transformers import SentenceTransformer
import pandas as pd
import argparse
import psycopg2

class SchemaVectorizer:
    def __init__(self, db_path, db_type="sqlite", vector_db_path="schema_vectors"):
        self.db_path = db_path
        self.db_type = db_type
        self.vector_db_path = vector_db_path
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.db = lancedb.connect(self.vector_db_path)

    def get_schema_descriptions(self):
        """Extracts schema information and returns a list of descriptive strings."""
        if self.db_type == "sqlite":
            return self._get_sqlite_schema_descriptions()
        elif self.db_type == "postgresql":
            return self._get_postgresql_schema_descriptions()
        else:
            raise ValueError("Unsupported database type")

    def _get_sqlite_schema_descriptions(self):
        descriptions = []
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            for table in tables:
                if table.startswith('sqlite_'):
                    continue
                cursor.execute(f"PRAGMA table_info(`{table}`);")
                columns = cursor.fetchall()
                for col in columns:
                    descriptions.append(f"Table: {table}, Column: {col[1]}, Type: {col[2]}")
            conn.close()
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
        return descriptions

    def _get_postgresql_schema_descriptions(self):
        descriptions = []
        table_descriptions = []
        column_descriptions = []
        try:
            print(f"Connecting to PostgreSQL database: {self.db_path}")
            conn = psycopg2.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all tables in the public schema
            print("Querying for tables in public schema...")
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            """)
            tables = [row[0] for row in cursor.fetchall()]
            print(f"Found {len(tables)} tables: {tables}")
            
            # If no tables found in public schema, try to list all schemas
            if not tables:
                print("No tables found in public schema. Checking available schemas...")
                cursor.execute("SELECT schema_name FROM information_schema.schemata")
                schemas = [row[0] for row in cursor.fetchall()]
                print(f"Available schemas: {schemas}")
                
                # Try to find tables in any schema
                cursor.execute("""
                    SELECT table_schema, table_name 
                    FROM information_schema.tables 
                    WHERE table_type = 'BASE TABLE'
                """)
                all_tables = cursor.fetchall()
                print(f"All tables in database: {all_tables}")
                
                # Look specifically for tables in the oncomx_v1_0_25 schema
                if 'oncomx_v1_0_25' in schemas:
                    print("Found oncomx_v1_0_25 schema. Getting tables from this schema...")
                    cursor.execute("""
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_schema = 'oncomx_v1_0_25' AND table_type = 'BASE TABLE'
                    """)
                    oncomx_tables = [row[0] for row in cursor.fetchall()]
                    print(f"Found {len(oncomx_tables)} tables in oncomx_v1_0_25 schema: {oncomx_tables}")
                    
                    # Process tables from the oncomx_v1_0_25 schema
                    for table in oncomx_tables:
                        print(f"Processing table from oncomx_v1_0_25 schema: {table}")
                        # Get column information
                        cursor.execute("""
                            SELECT column_name, data_type, is_nullable, column_default
                            FROM information_schema.columns
                            WHERE table_schema = 'oncomx_v1_0_25' AND table_name = %s
                            ORDER BY ordinal_position
                        """, (table,))
                        columns = cursor.fetchall()
                        print(f"Found {len(columns)} columns in table {table}")
                        
                        # Get primary key information
                        cursor.execute("""
                            SELECT c.column_name
                            FROM information_schema.table_constraints tc
                            JOIN information_schema.constraint_column_usage AS ccu USING (constraint_schema, constraint_name)
                            JOIN information_schema.columns AS c ON c.table_schema = tc.constraint_schema
                              AND tc.table_name = c.table_name AND ccu.column_name = c.column_name
                            WHERE constraint_type = 'PRIMARY KEY' AND tc.table_schema = 'oncomx_v1_0_25' AND tc.table_name = %s
                        """, (table,))
                        pks = cursor.fetchall()
                        pk_cols = [pk[0] for pk in pks] if pks else []
                        
                        # Get foreign key information
                        cursor.execute("""
                            SELECT
                                kcu.column_name,
                                ccu.table_name AS foreign_table_name,
                                ccu.column_name AS foreign_column_name
                            FROM
                                information_schema.table_constraints AS tc
                                JOIN information_schema.key_column_usage AS kcu
                                  ON tc.constraint_name = kcu.constraint_name
                                  AND tc.table_schema = kcu.table_schema
                                JOIN information_schema.constraint_column_usage AS ccu
                                  ON ccu.constraint_name = tc.constraint_name
                                  AND ccu.table_schema = tc.table_schema
                            WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = 'oncomx_v1_0_25' AND tc.table_name = %s
                        """, (table,))
                        fks = cursor.fetchall()
                        
                        # Create a comprehensive table description
                        column_names = [col[0] for col in columns]
                        table_desc = f"Table: oncomx_v1_0_25.{table}. Columns: {', '.join(column_names)}"
                        if pk_cols:
                            table_desc += f". Primary Key: {', '.join(pk_cols)}"
                        if fks:
                            fk_desc = []
                            for fk in fks:
                                col, ref_table, ref_col = fk
                                fk_desc.append(f"{col} references {ref_table}({ref_col})")
                            table_desc += f". Foreign Keys: {'; '.join(fk_desc)}"
                        
                        table_descriptions.append(table_desc)
                        
                        # Create detailed descriptions for each column
                        for col in columns:
                            col_name, data_type, is_nullable, default = col
                            nullable_str = "NULL" if is_nullable == "YES" else "NOT NULL"
                            default_str = f" DEFAULT {default}" if default else ""
                            
                            # Check if this column is a primary key
                            pk_status = "Primary Key" if col_name in pk_cols else ""
                            
                            # Check if this column is a foreign key
                            fk_status = ""
                            for fk in fks:
                                if fk[0] == col_name:
                                    fk_status = f"Foreign Key to {fk[1]}({fk[2]})"
                                    break
                            
                            # Create a comprehensive column description
                            col_desc = f"Table: oncomx_v1_0_25.{table}, Column: {col_name}, Type: {data_type}, {nullable_str}{default_str}"
                            if pk_status:
                                col_desc += f", {pk_status}"
                            if fk_status:
                                col_desc += f", {fk_status}"
                            
                            column_descriptions.append(col_desc)
                        
                        # Add sample data if possible (limited to 5 rows)
                        try:
                            cursor.execute(f"SELECT * FROM oncomx_v1_0_25.\"{table}\" LIMIT 5")
                            rows = cursor.fetchall()
                            if rows:
                                sample_data = f"Sample data for oncomx_v1_0_25.{table}:"
                                column_descriptions.append(sample_data)
                        except Exception as e:
                            print(f"Error getting sample data for oncomx_v1_0_25.{table}: {e}")
            
            # For each table, get its columns and data types
            for table in tables:
                print(f"Processing table: {table}")
                # Get column information
                cursor.execute("""
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = %s
                    ORDER BY ordinal_position
                """, (table,))
                columns = cursor.fetchall()
                print(f"Found {len(columns)} columns in table {table}")
                
                # Get primary key information
                cursor.execute("""
                    SELECT c.column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.constraint_column_usage AS ccu USING (constraint_schema, constraint_name)
                    JOIN information_schema.columns AS c ON c.table_schema = tc.constraint_schema
                      AND tc.table_name = c.table_name AND ccu.column_name = c.column_name
                    WHERE constraint_type = 'PRIMARY KEY' AND tc.table_name = %s
                """, (table,))
                pks = cursor.fetchall()
                pk_cols = [pk[0] for pk in pks] if pks else []
                
                # Get foreign key information
                cursor.execute("""
                    SELECT
                        kcu.column_name,
                        ccu.table_name AS foreign_table_name,
                        ccu.column_name AS foreign_column_name
                    FROM
                        information_schema.table_constraints AS tc
                        JOIN information_schema.key_column_usage AS kcu
                          ON tc.constraint_name = kcu.constraint_name
                          AND tc.table_schema = kcu.table_schema
                        JOIN information_schema.constraint_column_usage AS ccu
                          ON ccu.constraint_name = tc.constraint_name
                          AND ccu.table_schema = tc.table_schema
                    WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = %s
                """, (table,))
                fks = cursor.fetchall()
                
                # Create a comprehensive table description
                column_names = [col[0] for col in columns]
                table_desc = f"Table: {table}. Columns: {', '.join(column_names)}"
                if pk_cols:
                    table_desc += f". Primary Key: {', '.join(pk_cols)}"
                if fks:
                    fk_desc = []
                    for fk in fks:
                        col, ref_table, ref_col = fk
                        fk_desc.append(f"{col} references {ref_table}({ref_col})")
                    table_desc += f". Foreign Keys: {'; '.join(fk_desc)}"
                
                table_descriptions.append(table_desc)
                
                # Create detailed descriptions for each column
                for col in columns:
                    col_name, data_type, is_nullable, default = col
                    nullable_str = "NULL" if is_nullable == "YES" else "NOT NULL"
                    default_str = f" DEFAULT {default}" if default else ""
                    
                    # Check if this column is a primary key
                    pk_status = "Primary Key" if col_name in pk_cols else ""
                    
                    # Check if this column is a foreign key
                    fk_status = ""
                    for fk in fks:
                        if fk[0] == col_name:
                            fk_status = f"Foreign Key to {fk[1]}({fk[2]})"
                            break
                    
                    # Create a comprehensive column description
                    col_desc = f"Table: {table}, Column: {col_name}, Type: {data_type}, {nullable_str}{default_str}"
                    if pk_status:
                        col_desc += f", {pk_status}"
                    if fk_status:
                        col_desc += f", {fk_status}"
                    
                    column_descriptions.append(col_desc)
                
                # Add sample data if possible (limited to 5 rows)
                try:
                    cursor.execute(f"SELECT * FROM \"{table}\" LIMIT 5")
                    rows = cursor.fetchall()
                    if rows:
                        sample_data = f"Sample data for {table}:"
                        column_descriptions.append(sample_data)
                except Exception as e:
                    print(f"Error getting sample data for {table}: {e}")
            
            # Combine all descriptions
            descriptions = table_descriptions + column_descriptions
            
            conn.close()
        except psycopg2.Error as e:
            print(f"PostgreSQL error: {e}")
            import traceback
            traceback.print_exc()
            
            # Try with a different connection approach as a fallback
            try:
                print("Trying alternative connection approach...")
                import os
                # Extract connection parameters from the connection string
                # Format: postgresql://user:password@host:port/dbname
                parts = self.db_path.split('://')[-1].split('@')
                user_pass = parts[0].split(':')
                host_port_db = parts[1].split('/')
                
                user = user_pass[0]
                password = user_pass[1] if len(user_pass) > 1 else ''
                
                host_port = host_port_db[0].split(':')
                host = host_port[0]
                port = host_port[1] if len(host_port) > 1 else '5432'
                
                dbname = host_port_db[1] if len(host_port_db) > 1 else ''
                
                # Set environment variables for psycopg2
                os.environ['PGUSER'] = user
                os.environ['PGPASSWORD'] = password
                os.environ['PGHOST'] = host
                os.environ['PGPORT'] = port
                os.environ['PGDATABASE'] = dbname
                
                print(f"Connecting with parameters: user={user}, host={host}, port={port}, dbname={dbname}")
                conn = psycopg2.connect(
                    dbname=dbname,
                    user=user,
                    password=password,
                    host=host,
                    port=port
                )
                
                cursor = conn.cursor()
                
                # Get all tables in the public schema
                print("Querying for tables with alternative connection...")
                cursor.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                """)
                tables = [row[0] for row in cursor.fetchall()]
                print(f"Found {len(tables)} tables with alternative connection: {tables}")
                
                # Add a basic description for each table
                for table in tables:
                    table_descriptions.append(f"Table: {table}")
                    
                    # Get column information
                    cursor.execute("""
                        SELECT column_name, data_type
                        FROM information_schema.columns
                        WHERE table_schema = 'public' AND table_name = %s
                    """, (table,))
                    columns = cursor.fetchall()
                    
                    for col in columns:
                        column_descriptions.append(f"Table: {table}, Column: {col[0]}, Type: {col[1]}")
                
                # Combine all descriptions
                descriptions = table_descriptions + column_descriptions
                conn.close()
                
            except Exception as alt_e:
                print(f"Alternative connection approach failed: {alt_e}")
                traceback.print_exc()
        
        if not descriptions:
            print("WARNING: No schema descriptions were generated. This will result in an empty vector database.")
            # Add a placeholder description to avoid completely empty results
            descriptions = ["Database schema could not be extracted. Please check the connection parameters and database access."]
            
        return descriptions

    def vectorize_and_store_schema(self):
        """Vectorizes schema descriptions and stores them in LanceDB."""
        descriptions = self.get_schema_descriptions()
        if not descriptions:
            print("No schema descriptions found.")
            return

        # Create embeddings for each description
        print(f"Generating embeddings for {len(descriptions)} schema descriptions...")
        embeddings = self.model.encode(descriptions, convert_to_tensor=True)
        
        # Extract table name from the path or connection string
        if self.db_type == 'postgresql':
            # Extract database name from connection string
            # Format is typically: postgresql://user:password@host:port/dbname
            try:
                table_name = self.db_path.split('/')[-1]
                # Handle case where connection string doesn't end with dbname
                if not table_name or '?' in table_name:
                    # Try to extract from standard connection string format
                    import re
                    match = re.search(r'/([^/?]+)(\?|$)', self.db_path)
                    if match:
                        table_name = match.group(1)
                    else:
                        # Fallback to a default name if we can't extract it
                        table_name = "oncomx_v1_0_25_small"  # Using the specific DB name from the error
            except Exception as e:
                print(f"Error extracting database name from connection string: {e}")
                table_name = "postgresql_db"
        else:
            # For SQLite, use the basename without extension
            table_name = os.path.splitext(os.path.basename(self.db_path))[0]

        # Prepare data for LanceDB
        data = pd.DataFrame({
            "vector": [e.numpy().tolist() for e in embeddings],
            "text": descriptions,
            # Add metadata to help with filtering and retrieval
            "db_type": [self.db_type] * len(descriptions),
            "description_type": ["schema"] * len(descriptions)
        })

        try:
            # Create or overwrite the table in LanceDB
            self.db.create_table(table_name, data=data, mode="overwrite")
            print(f"Successfully created and populated table '{table_name}' in LanceDB with {len(descriptions)} schema descriptions.")
            
            # Create a directory to store a backup of the schema descriptions
            schema_backup_dir = os.path.join("schema_backups", table_name)
            os.makedirs(schema_backup_dir, exist_ok=True)
            
            # Save the schema descriptions to a text file for reference
            schema_backup_file = os.path.join(schema_backup_dir, "schema_descriptions.txt")
            with open(schema_backup_file, 'w', encoding='utf-8') as f:
                f.write(f"Schema descriptions for {table_name} ({self.db_type}):\n\n")
                for desc in descriptions:
                    f.write(f"{desc}\n")
            
            print(f"Schema descriptions backed up to {schema_backup_file}")
            
        except Exception as e:
            print(f"Error creating table in LanceDB: {e}")
            import traceback
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Vectorize a database schema and store it in LanceDB.")
    parser.add_argument('--db_type', required=True, choices=['sqlite', 'postgresql'])
    parser.add_argument('--db_path', required=True, help='Path to the database file or connection string.')
    parser.add_argument('--vector_db_path', default='schema_vectors', help='Path to the LanceDB vector store.')
    args = parser.parse_args()

    vectorizer = SchemaVectorizer(args.db_path, args.db_type, args.vector_db_path)
    vectorizer.vectorize_and_store_schema()

if __name__ == "__main__":
    main()
