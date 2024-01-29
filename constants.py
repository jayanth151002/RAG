import os
from dotenv import load_dotenv

load_dotenv()

openai_key = os.getenv("OPENAI_KEY")

file_path = "gpt4.pdf"

prompt_template = """
This is the user query that you'll have to answer:
{user_query}
This is the additional context that you can use to answer the query:
{context}
You should not directly answer the query, but instead answer the query using the context.
Do not state facts directly from the context, you have to comprehend the context and then answer the query in your own words. You are not allowed to say that you have been given the additional context.
"""
