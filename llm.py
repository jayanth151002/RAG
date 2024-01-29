from openai import OpenAI
from constants import openai_key, prompt_template

client = OpenAI(
    api_key=openai_key,
)


class Llm:
    def get_response(self, user_query, sim_chunks):
        chunk_string = "\n".join([f"<chunk>\n{c}\n</chunk>" for c in sim_chunks])
        prompt = prompt_template.format(
            user_query=user_query,
            context=chunk_string
        )
        result = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-3.5-turbo",
        )
        response = result.choices[0].message.content
        return response
