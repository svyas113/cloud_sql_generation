import json
import os
import argparse
import asyncio
from dotenv import load_dotenv
from deepseek_sql_generator_finalV5 import process_dataset

async def main():
    # Load environment variables
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='Batch process SQL generation for multiple queries')
    parser.add_argument('--api_key', default=os.getenv('API_KEY'), help='API key (default: from .env)')
    parser.add_argument('--input_file', required=True, help='Input JSON file with questions')
    parser.add_argument('--output_file', required=True, help='Output JSON file for results')
    parser.add_argument('--db_connection_string', default=os.getenv('DB_CONNECTION_STRING'), 
                        help='Database connection string (default: from .env)')
    parser.add_argument('--model', default='deepseek/deepseek-chat', help='Model to use')
    parser.add_argument('--batch_size', type=int, default=50, help='Number of questions per batch')
    parser.add_argument('--checkpoint_file', default='checkpoint.json', help='File to store checkpoint information')
    args = parser.parse_args()
    
    # Validate required parameters
    if not args.api_key:
        print("Error: API_KEY must be set via --api_key or in .env file")
        return
        
    if not args.db_connection_string:
        print("Error: DB_CONNECTION_STRING must be set via --db_connection_string or in .env file")
        return
    
    # Load questions
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            all_questions = json.load(f)
        print(f"Loaded {len(all_questions)} questions from {args.input_file}")
    except Exception as e:
        print(f"Error loading input file: {e}")
        return
    
    # Load checkpoint if exists
    start_idx = 0
    if os.path.exists(args.checkpoint_file):
        try:
            with open(args.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
                start_idx = checkpoint.get('last_processed_idx', 0) + 1
                print(f"Resuming from checkpoint: starting at index {start_idx}")
        except Exception as e:
            print(f"Error loading checkpoint file: {e}")
            print("Starting from the beginning")
    
    # Process in batches
    total_questions = len(all_questions)
    all_results = []
    
    for batch_start in range(start_idx, total_questions, args.batch_size):
        batch_end = min(batch_start + args.batch_size, total_questions)
        batch = all_questions[batch_start:batch_end]
        
        print(f"\n{'='*50}")
        print(f"Processing batch {batch_start+1}-{batch_end} of {total_questions}")
        print(f"{'='*50}")
        
        # Create a temporary file for this batch
        batch_file = f"{args.output_file}.part{batch_start}"
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(batch, f, indent=2)
        
        # Process batch
        try:
            results, _ = await process_dataset(
                api_key=args.api_key,
                input_file=batch_file,
                output_file=f"{batch_file}.results",
                db_dir=None,
                db_connection_string=args.db_connection_string,
                model=args.model,
                db_type='postgresql',
                feedback_loop=True
            )
            
            # Load results from the output file
            with open(f"{batch_file}.results", 'r', encoding='utf-8') as f:
                batch_results = json.load(f)
                all_results.extend(batch_results)
            
            # Save all results so far
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2)
            
            # Save checkpoint
            with open(args.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump({'last_processed_idx': batch_end - 1}, f)
            
            print(f"Processed {batch_end}/{total_questions} questions")
            
            # Clean up temporary files
            try:
                os.remove(batch_file)
                os.remove(f"{batch_file}.results")
            except:
                pass
                
        except Exception as e:
            print(f"Error processing batch: {e}")
            import traceback
            traceback.print_exc()
            
            # Save checkpoint at the last successful batch
            with open(args.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump({'last_processed_idx': batch_start - 1}, f)
            
            print(f"Checkpoint saved at index {batch_start - 1}")
            break
    
    print(f"\nProcessing complete. Processed {len(all_results)} questions.")
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    asyncio.run(main())
