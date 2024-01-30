from constants import file_path
from pathlib import Path
from reader import PdfReader
from tokenizer import Tokenizer
from encoder import Encoder
from llm import Llm


def get_answer(query, pdf_file_path):
    pdf_content = PdfReader().load_data(Path(pdf_file_path))
    chunks = Tokenizer.tokenize_and_chunk(pdf_content)
    enc = Encoder()
    enc.load_embeddings(chunks)

    sim_chunks = enc.similarity_search(query)
    response = Llm().get_response(query, sim_chunks)
    return response


query = "What are some of the advanced capabilities of the latest Gemini model?"
ans_without_rag = Llm().get_response(query)
print("Answer without RAG:\n\n", ans_without_rag, "\n\n")
ans_with_rag = get_answer(query, file_path)
print("Answer with RAG:\n\n", ans_with_rag)