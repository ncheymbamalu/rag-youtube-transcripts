"""This module contains functionality for implementing RAG over YouTube video transcripts."""

from pathlib import Path

import polars as pl
from groq.types.chat import ChatCompletion, ChatCompletionMessage
from omegaconf import DictConfig

from rag_youtube_transcripts.config import Config
from rag_youtube_transcripts.utils import GROQ_CLIENT, get_semantic_search_results


PARAMS: DictConfig = Config.load_params(Path(__file__).stem)


def create_user_prompt(query: str) -> str:
    """Creates the RAG system's user prompt.

    Args:
        query (str): Input query.

    Returns:
        str: User prompt.
    """
    try:
        user_prompt: str = PARAMS.user_prompt
        delimiter: str = PARAMS.delimiter
        results: pl.DataFrame = get_semantic_search_results(query)
        context: str = f"\n{delimiter}\n".join(
            f"TITLE: {record.get('title')}\n"
            f"URL: {record.get('url')}\n"
            f"START: {record.get('start')}\n"
            f"END: {record.get('end')}\n"
            f"EXCERPT: {record.get('excerpt')}"
            for record in results.to_dicts()
        )
        return user_prompt.format(context=context, query=query)
    except Exception as e:
        raise e


def generate_response(
    query: str,
    llm: str = PARAMS.llm,
    temperature: float | int = PARAMS.temperature,
    max_completion_tokens: int = PARAMS.max_output_tokens,
    reasoning_effort: str = PARAMS.reasoning_effort
) -> str:
    """Generates a response to the input query.

    Args:
        query (str): Input query.
        llm (str, optional): The LLM used for RAG. Defaults to PARAMS.llm.
        temperature (float | int, optional): Parameter between 0 and 2 inclusive,
        that controls the randomness of the response. The lower the temperature,
        the more repetitive the response. Defaults to PARAMS.temperature.
        max_completion_tokens (int, optional): Maximum number of tokens used to create
        the response. Defaults to PARAMS.max_output_tokens.
        reasoning_effort (str, optional): Controls how many internal reasoning tokens the
        llm generates before producing its response. Defaults to PARAMS.reasoning_effort

    Returns:
        str: Response.
    """
    try:
        system_prompt: str = PARAMS.system_prompt
        system_prompt = system_prompt.format(
            delimiter=PARAMS.delimiter,
            youtube_search_url=PARAMS.youtube_search_url,
            query=query
        )
        user_prompt: str = create_user_prompt(query)
        completion: ChatCompletion = GROQ_CLIENT.chat.completions.create(
            model=llm,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort
        )
        message: ChatCompletionMessage = completion.choices[0].message
        return message.content
    except Exception as e:
        raise e
