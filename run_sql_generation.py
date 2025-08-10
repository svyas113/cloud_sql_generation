import os
import argparse
import asyncio
from dotenv import load_dotenv
from deepseek_sql_generator_finalV5 import process_dataset

async def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate SQL queries using DeepSeek API via litellm')
    parser.add_argument('--api_key', default=os.getenv('API_KEY'), help='DeepSeek API key (default: from .env)')
    parser.add_argument('--input_file', required=True, help='Input JSON file with questions')
    parser.add_argument('--output_file', required=True, help='Output JSON file to save detailed results')
    parser.add_argument('--db_connection_string', default=os.getenv('DB_CONNECTION_STRING'), 
                        help='Database connection string for PostgreSQL (default: from .env)')
    parser.add_argument('--model', default='deepseek/deepseek-chat',
                        help="Model to use via litellm (default: 'deepseek/deepseek-chat')")
    parser.add_argument('--limit', type=int, help='Limit number of questions to process (for testing)')
    parser.add_argument('--feedback_loop', action='store_true', default=True,
                        help='Enable feedback loop for SQL execution and correction')
    parser.add_argument('--no_feedback_loop', action='store_false', dest='feedback_loop',
                        help='Disable feedback loop for SQL execution and correction')
    parser.add_argument('--excel_output', default='feedback_results.xlsx',
                        help='Path to save feedback results to Excel file')

    args = parser.parse_args()

    # Validate required parameters
    if not args.api_key:
        print("Error: API key is required. Set via --api_key or API_KEY in .env file.")
        return
        
    if not args.db_connection_string:
        print("Error: Database connection string is required. Set via --db_connection_string or DB_CONNECTION_STRING in .env file.")
        return

    print("Starting SQL generation process...")
    print(f"Using model: {args.model}")
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Feedback loop: {'Enabled' if args.feedback_loop else 'Disabled'}")
    if args.feedback_loop:
        print(f"Excel output: {args.excel_output}")
    if args.limit:
        print(f"Processing limit: {args.limit} questions")

    await process_dataset(
        api_key=args.api_key,
        input_file=args.input_file,
        output_file=args.output_file,
        db_dir=None,
        db_connection_string=args.db_connection_string,
        model=args.model,
        db_type='postgresql',
        limit=args.limit,
        feedback_loop=args.feedback_loop,
        excel_output=args.excel_output if args.feedback_loop else None
    )
    print("SQL generation process finished.")

if __name__ == "__main__":
    asyncio.run(main())
