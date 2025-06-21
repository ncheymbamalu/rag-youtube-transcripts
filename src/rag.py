"""This module contains functionality for implementing RAG across YouTube video transcripts."""

import polars as pl

from groq.types.chat import ChatCompletion, ChatCompletionMessage

from src.config import Config
from src.utils import get_semantic_search_results, groq_client

system_prompt: str = """\
Craft a response to the query based on the provided information.

The provided information may contain the TITLE and URL of one or more YouTube videos, each \
separated by {delimiter}.

If the provided information contains URLs, ALWAYS include them in your response.

If the provided information is insufficient for crafting a response to the query, provide useful \
information from your pre-existing knowledge base, but DO NOT hallucinate or make things up. In \
addition, include a relevant YouTube link via `{youtube_search_url}={query}`.

Your response should be properly formatted and easy to read.\
"""


def create_user_prompt(query: str) -> str:
    """Creates the RAG system's user prompt.

    Args:
        query (str): Input query.

    Returns:
        str: User prompt.
    """
    try:
        delimiter: str = Config.load_params("delimiter")
        results: pl.DataFrame = get_semantic_search_results(query)
        context: str = f"\n{delimiter}\n".join(
            f"TITLE: {record.get('title')}\nURL: {record.get('url')}"
            for record in results.to_dicts()
        )
        user_prompt: str = f"""\
Use the following information:

```
{context}
```

to respond to the query: {query}\
        """
        return user_prompt
    except Exception as e:
        raise e


def generate_response(
    query: str,
    llm: str = Config.load_params("llm").get("rag"),
    temperature: float | int = Config.load_params("temperature").get("rag"),
    max_completion_tokens: int = Config.load_params("max_output_tokens").get("rag")
) -> str:
    """Generates a response to the input query.

    Args:
        query (str): Input query.
        llm (str, optional): The LLM used for RAG.
        Defaults to Config.load_params("llm").get("rag").
        temperature (float | int, optional): Parameter between 0 and 2, inclusive, that
        contols the randomness of the response. The lower the temperature, the more repetitive
        the response. Config.load_params("temperature").get("rag").
        max_completion_tokens (int, optional): Maximum number of tokens used to create
        the response. Config.load_params("max_output_tokens").get("rag").

    Returns:
        str: Response.
    """
    try:
        completion: ChatCompletion = groq_client.chat.completions.create(
            model=llm,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt.format(
                        delimiter=Config.load_params("delimiter"),
                        youtube_search_url=Config.load_params("youtube_search_url"),
                        query=query
                    )
                },
                {
                    "role": "user",
                    "content": create_user_prompt(query)
                }
            ],
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )
        message: ChatCompletionMessage = completion.choices[0].message
        return message.content
    except Exception as e:
        raise e
