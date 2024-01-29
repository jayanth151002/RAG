from constants import file_path
from pathlib import Path
from reader import PdfReader
from tokenizer import Tokenizer
from encoder import Encoder
from llm import Llm


if __name__ == "__main__":
    pdf_content = PdfReader().load_data(Path(file_path))
    chunks = Tokenizer.tokenize_and_chunk(pdf_content)
    enc = Encoder()
    enc.load_embeddings(chunks)

    query = "What are some of the advanced capabilities of the latest GPT-4 model?"
    sim_chunks = enc.similarity_search(query)
    response = Llm().get_response(query, sim_chunks)
    print(response)
