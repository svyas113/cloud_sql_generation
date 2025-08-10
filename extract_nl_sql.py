import json

def extract_nl_sql_pairs_from_multiple_files(input_files, output_json_path):
    """
    Reads multiple JSON files containing ScienceBenchmark data, extracts the
    natural language questions and their corresponding SQL queries,
    and saves them to a single combined JSON file.

    Args:
        input_files (list): A list of paths to the input JSON files.
        output_json_path (str): The path to the output JSON file to be created.
    """
    # A list to hold all the extracted question-SQL pair objects from all files
    all_extracted_data = []
    
    # Process each input file
    for json_file_path in input_files:
        try:
            print(f"Processing file: {json_file_path}")
            # Open and load the input JSON file
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            file_extracted_data = []
            # Iterate through each entry in the JSON data
            for entry in data:
                # Get the question and the full SQL query string
                question = entry.get('question')
                sql_query = entry.get('query')

                # Ensure both question and query exist before adding
                if question and sql_query:
                    file_extracted_data.append({
                        'question': question,
                        'sql': sql_query,
                        'source_file': json_file_path  # Optional: track which file it came from
                    })

            print(f"Successfully extracted {len(file_extracted_data)} question-SQL pairs from {json_file_path}.")
            
            # Add the extracted data from this file to the combined list
            all_extracted_data.extend(file_extracted_data)

        except FileNotFoundError:
            print(f"Error: The file '{json_file_path}' was not found.")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from the file '{json_file_path}'.")
        except Exception as e:
            print(f"An unexpected error occurred with file {json_file_path}: {e}")
    
    # Write the combined extracted data to a single output JSON file
    try:
        with open(output_json_path, 'w', encoding='utf-8') as json_out_file:
            json.dump(all_extracted_data, json_out_file, ensure_ascii=False, indent=4)
        
        print(f"Combined data successfully saved to {output_json_path}")
        print(f"Total question-SQL pairs extracted: {len(all_extracted_data)}")
    except Exception as e:
        print(f"Error saving combined data to {output_json_path}: {e}")

# --- --- --- --- --- ---
# How to use the script
# --- --- --- --- --- ---

# List of input files to process
input_files = ['dev.json', 'seed.json', 'synth.json']

# Output file for the combined data
output_file = 'combined_question_sql_pairs.json'

# Run the function to process all files and combine the results
extract_nl_sql_pairs_from_multiple_files(input_files, output_file)
