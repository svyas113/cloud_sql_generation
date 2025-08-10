import lancedb
from sentence_transformers import SentenceTransformer
import litellm
import argparse
import os

class HydeModule:
    def __init__(self, api_key, model="deepseek/deepseek-chat", vector_db_path="schema_vectors"):
        self.api_key = api_key
        self.model = model
        self.vector_db_path = vector_db_path
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.db = lancedb.connect(self.vector_db_path)

    async def generate_hypothetical_document(self, question):
        """Generates a hypothetical document based on the user's question."""
        prompt = f"Please generate a descriptive answer for the following question. This answer should describe the ideal data needed to answer the question, but should not be the answer itself.\n\nQuestion: {question}"
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                api_key=self.api_key,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating hypothetical document: {e}")
            return None

    def search_schema(self, query_embedding, table_name, limit=25):
        """Searches the vectorized schema for the most relevant descriptions."""
        try:
            # Try to open the table with the given name
            try:
                tbl = self.db.open_table(table_name)
            except Exception as first_error:
                print(f"Warning: Could not open table '{table_name}': {first_error}")
                
                # If the table_name might be from a PostgreSQL connection string, try to extract the database name
                if '://' in table_name:
                    try:
                        # Extract database name from connection string
                        import re
                        match = re.search(r'/([^/?]+)(\?|$)', table_name)
                        if match:
                            extracted_name = match.group(1)
                            print(f"Trying with extracted database name: {extracted_name}")
                            tbl = self.db.open_table(extracted_name)
                        else:
                            # Fallback to a hardcoded name if we can't extract it
                            fallback_name = "oncomx_v1_0_25_small"
                            print(f"Falling back to hardcoded database name: {fallback_name}")
                            tbl = self.db.open_table(fallback_name)
                    except Exception as second_error:
                        print(f"Error with fallback approach: {second_error}")
                        return []
                else:
                    # If it's not a connection string, we don't have a fallback
                    return []
            
            # Search the table
            results = tbl.search(query_embedding).limit(limit).to_df()
            return results['text'].tolist()
        except Exception as e:
            print(f"Error searching schema: {e}")
            import traceback
            traceback.print_exc()
            return []

    async def get_focused_schema(self, question, db_id, limit=5):
        """Orchestrates the HyDE process to get a focused schema context."""
        hypothetical_doc = await self.generate_hypothetical_document(question)
        if not hypothetical_doc:
            return ""

        query_embedding = self.embedding_model.encode(hypothetical_doc)
        
        # The table name in LanceDB corresponds to the db_id
        focused_schema_parts = self.search_schema(query_embedding, db_id, limit=limit)
        
        return "\n".join(focused_schema_parts)
        
    async def get_focused_schema_with_error(self, question, db_id, failed_sql, error_message):
        """
        Generate a focused schema context based on a question, failed SQL, and error message.
        
        Args:
            question: The natural language question
            db_id: Database identifier
            failed_sql: The SQL query that failed
            error_message: The error message from the database
            
        Returns:
            A focused schema context
        """
        # Create a prompt that includes the failed SQL and error message
        enhanced_prompt = f"""
        I need to generate a SQL query for the following question: {question}
        
        My previous attempt failed with this SQL:
        {failed_sql}
        
        The error message was:
        {error_message}
        
        Please generate a hypothetical document describing the ideal database schema needed to answer this question correctly.
        Focus on the tables and columns that would be required, especially addressing the issues that caused the error.
        """
        
        # Generate a hypothetical document based on the enhanced prompt
        hypothetical_doc = await self.generate_hypothetical_document(enhanced_prompt)
        if not hypothetical_doc:
            return ""

        # Encode and search for relevant schema parts
        query_embedding = self.embedding_model.encode(hypothetical_doc)
        focused_schema_parts = self.search_schema(query_embedding, db_id)
        
        return "\n".join(focused_schema_parts)

async def main():
    parser = argparse.ArgumentParser(description="Use HyDE to get a focused schema context.")
    parser.add_argument('--api_key', required=True, help='API key for the LLM.')
    parser.add_argument('--question', required=True, help='The natural language question.')
    parser.add_argument('--db_id', required=True, help='The ID of the database to search.')
    parser.add_argument('--vector_db_path', default='schema_vectors', help='Path to the LanceDB vector store.')
    args = parser.parse_args()

    hyde = HydeModule(args.api_key, vector_db_path=args.vector_db_path)
    focused_schema = await hyde.get_focused_schema(args.question, args.db_id)
    
    print("Focused Schema Context:")
    print(focused_schema)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
