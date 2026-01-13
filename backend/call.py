import logging

from utils.llm_caller import LLM
from backend.backend_utils import PromptType, opt_messages_to_list
from utils.config_mcts import Config

logger = logging.getLogger("ml-master")

def r1_query(
    prompt: PromptType | None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    cfg:Config=None,
    **model_kwargs,
):
    llm = LLM(
        base_url=cfg.agent.code.base_url,
        api_key=cfg.agent.code.api_key,
        model_name=cfg.agent.code.model
    )
    logger.info(f"using {llm.model_name} to generate code.")
    logger.info("---Querying model---", extra={"verbose": True})
    if type(prompt) == str:
        logger.info(f"prompt: {prompt}", extra={"verbose": True})
    else:
        try:
            if isinstance(prompt, (list, tuple)) and len(prompt) >= 2:
                logger.info(f"prompt: {prompt[0].get('content', prompt[0])}\n{prompt[1].get('content', prompt[1])}", extra={"verbose": True})
            else:
                logger.info(f"prompt: {prompt}", extra={"verbose": True})
        except (KeyError, TypeError, IndexError):
            logger.info(f"prompt: {prompt}", extra={"verbose": True})
    
    # Ensure prompt is in the correct message list format
    if isinstance(prompt, str):
        messages = opt_messages_to_list(None, prompt)
    elif isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], dict) and "role" in prompt[0]:
        # Already in correct format
        messages = prompt
    elif isinstance(prompt, list) and len(prompt) >= 2:
        # Assume first element is system message, second is user message
        messages = opt_messages_to_list(prompt[0] if isinstance(prompt[0], str) else str(prompt[0]), 
                                       prompt[1] if isinstance(prompt[1], str) else str(prompt[1]))
    else:
        # Fallback: treat as user message
        messages = opt_messages_to_list(None, str(prompt))
    
    model_kwargs = model_kwargs | {
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    if cfg.agent.steerable_reasoning == True:
        response = llm.stream_complete(
            messages,
            **model_kwargs
        )
        
    else:
        response = llm.stream_generate(
            messages,
            **model_kwargs
        )


    if "</think>" in response:
        res = response[response.find("</think>")+8:]
    else:
        res = response

    logger.info(f"response:\n{response}", extra={"verbose": True})
    logger.info(f"response without think:\n{res}", extra={"verbose": True})
    logger.info(f"---Query complete---", extra={"verbose": True})
    return res

def gpt_query(
    prompt: PromptType | None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    cfg:Config=None,
    **model_kwargs,
):
    llm = LLM(
        base_url=cfg.agent.code.base_url,
        api_key=cfg.agent.code.api_key,
        model_name=cfg.agent.code.model
    )
    logger.info(f"using {llm.model_name} to generate code.")
    logger.info("---Querying model---", extra={"verbose": True})
    if type(prompt) == str:
        logger.info(f"prompt: {prompt}", extra={"verbose": True})
    else:
        try:
            if isinstance(prompt, (list, tuple)) and len(prompt) >= 2:
                logger.info(f"prompt: {prompt[0].get('content', prompt[0])}\n{prompt[1].get('content', prompt[1])}", extra={"verbose": True})
            else:
                logger.info(f"prompt: {prompt}", extra={"verbose": True})
        except (KeyError, TypeError, IndexError):
            logger.info(f"prompt: {prompt}", extra={"verbose": True})
    
    # Ensure prompt is in the correct message list format
    if isinstance(prompt, str):
        messages = opt_messages_to_list(None, prompt)
    elif isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], dict) and "role" in prompt[0]:
        # Already in correct format
        messages = prompt
    elif isinstance(prompt, list) and len(prompt) >= 2:
        # Assume first element is system message, second is user message
        messages = opt_messages_to_list(prompt[0] if isinstance(prompt[0], str) else str(prompt[0]), 
                                       prompt[1] if isinstance(prompt[1], str) else str(prompt[1]))
    else:
        # Fallback: treat as user message
        messages = opt_messages_to_list(None, str(prompt))
    
    model_kwargs = model_kwargs | {
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    response = llm.stream_generate(
        messages,
        **model_kwargs
    )

    res = response
    logger.info(f"response:\n{response}", extra={"verbose": True})
    logger.info(f"---Query complete---", extra={"verbose": True})
    return res