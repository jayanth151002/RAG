import tiktoken

class Tokenizer:
    def tokenize_and_chunk(text, token_limit=256):
        encoding = tiktoken.get_encoding("cl100k_base")
        chunks = []
        tokens = encoding.encode(text)

        while tokens:
            chunk = tokens[:token_limit]
            chunk_text = encoding.decode(chunk)
            last_punctuation = max(
                chunk_text.rfind("."),
                chunk_text.rfind("?"),
                chunk_text.rfind("!"),
                chunk_text.rfind("\n"),
            )
            if last_punctuation != -1 and len(tokens) > token_limit:
                chunk_text = chunk_text[: last_punctuation + 1]
            cleaned_text = chunk_text.replace("\n", " ").strip()
            if cleaned_text and (not cleaned_text.isspace()):
                chunks.append(cleaned_text)
            tokens = tokens[len(encoding.encode(chunk_text)) :]

        return chunks
