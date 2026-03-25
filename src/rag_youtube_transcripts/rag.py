"""This module contains functionality for implementing RAG over YouTube video transcripts."""

import polars as pl
from groq.types.chat import ChatCompletion, ChatCompletionMessage

from rag_youtube_transcripts.config import Config
from rag_youtube_transcripts.utils import GROQ_CLIENT, get_semantic_search_results


SYSTEM_PROMPT: str = Config.load_params("rag_system_prompt")


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
            f"TITLE: {record.get('title')}\n"
            f"URL: {record.get('url')}\n"
            f"START: {record.get('start')}\n"
            f"END: {record.get('end')}"
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
    llm: str = Config.load_params("llm").rag,
    temperature: float | int = Config.load_params("temperature").rag,
    max_completion_tokens: int = Config.load_params("max_output_tokens").rag
) -> str:
    """Generates a response to the input query.

    Args:
        query (str): Input query.
        llm (str, optional): The LLM used for RAG.
        Defaults to Config.load_params("llm").rag.
        temperature (float | int, optional): Parameter between 0 and 2 inclusive,
        that controls the randomness of the response. The lower the temperature,
        the more repetitive the response. Defaults to Config.load_params("temperature").rag.
        max_completion_tokens (int, optional): Maximum number of tokens used to create
        the response. Defaults to Config.load_params("max_output_tokens").rag.

    Returns:
        str: Response.
    """
    try:
        completion: ChatCompletion = GROQ_CLIENT.chat.completions.create(
            model=llm,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT.format(
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
