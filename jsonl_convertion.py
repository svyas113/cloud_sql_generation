import json

def convert_json_to_jsonl(input_file_path, output_file_path, db_id='oncomx'):
    """
    Converts a JSON file containing a list of question-SQL pairs to a JSONL file.

    Each line in the output JSONL file will be a JSON object with a structure
    similar to the provided dev.jsonl example.

    Args:
        input_file_path (str): The path to the input JSON file.
        output_file_path (str): The path where the output JSONL file will be saved.
        db_id (str, optional): The database ID to use for all entries.
                               Defaults to 'your_db_id'.
    """
    try:
        # Open the input JSON file and load its content
        with open(input_file_path, 'r', encoding='utf-8') as f_in:
            data = json.load(f_in)

        # Open the output JSONL file for writing
        with open(output_file_path, 'w', encoding='utf-8') as f_out:
            # Iterate through each item in the input data with an index
            for i, item in enumerate(data):
                # Ensure the item has the expected 'question' and 'sql' keys
                if 'question' not in item or 'sql' not in item:
                    print(f"Skipping item at index {i} due to missing keys.")
                    continue

                # Create the new JSON object with the desired structure
                new_record = {
                    "question_id": i,
                    "db_id": db_id,
                    "question": item['question'],
                    "evidence": "",  # Set evidence to an empty string
                    "SQL": item['sql'],
                    "difficulty": "default" # Set a default difficulty
                }

                # Convert the new record to a JSON string and write it to the file
                # followed by a newline character to make it a valid JSONL file.
                f_out.write(json.dumps(new_record) + '\n')

        print(f"Successfully converted {input_file_path} to {output_file_path}")

    except FileNotFoundError:
        print(f"Error: The file '{input_file_path}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file '{input_file_path}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    # Define the input and output file paths
    # NOTE: Make sure 'question_sql_pairs.json' is in the same directory
    # as this script, or provide the full path to it.
    input_json_file = 'combined_question_sql_pairs.json'
    output_jsonl_file = 'converted_dev.jsonl'

    # Call the conversion function
    convert_json_to_jsonl(input_json_file, output_jsonl_file)
