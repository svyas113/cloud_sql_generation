import json

def convert_jsonl_to_sql_format(input_jsonl_path, output_file_path):
    """
    Converts a JSONL file into a tab-separated text file with SQL and db_id.

    Each line in the input file is expected to be a JSON object containing
    at least "SQL" and "db_id" keys. The output file will have lines
    formatted as: SQL_QUERY\tdb_id

    Args:
        input_jsonl_path (str): The path to the input JSONL file.
        output_file_path (str): The path where the output text file will be saved.
    """
    try:
        # Open the input JSONL file for reading
        with open(input_jsonl_path, 'r', encoding='utf-8') as f_in:
            # Open the output file for writing
            with open(output_file_path, 'w', encoding='utf-8') as f_out:
                # Process each line in the JSONL file
                for line in f_in:
                    # Skip empty lines
                    if not line.strip():
                        continue

                    try:
                        # Parse the JSON object from the line
                        record = json.loads(line)

                        # Extract the SQL query and the database ID
                        sql_query = record.get('SQL', '')
                        # Split by lines, strip whitespace, and join with a single space
                        sql_query = ' '.join(line.strip() for line in sql_query.splitlines() if line.strip())
                        db_id = record.get('db_id', '')

                        # Write the formatted string to the output file
                        # The format is SQL_QUERY<tab>db_id
                        f_out.write(f"{sql_query}\t{db_id}\n")

                    except json.JSONDecodeError:
                        print(f"Warning: Skipping a line that is not valid JSON: {line.strip()}")
                        continue

        print(f"Successfully converted {input_jsonl_path} to {output_file_path}")

    except FileNotFoundError:
        print(f"Error: The file '{input_jsonl_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    # Define the input and output file paths.
    # Assumes 'converted_dev.jsonl' (the output from the previous script)
    # is in the same directory.
    input_file = 'converted_dev.jsonl'
    output_file = 'gold.json'

    # Call the conversion function
    convert_jsonl_to_sql_format(input_file, output_file)
