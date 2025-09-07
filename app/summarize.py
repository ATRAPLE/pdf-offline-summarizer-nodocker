import os
import logging
import time
from typing import List
import requests

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

def _chat(model: str, messages: list, temperature: float = 0.1, max_retries: int = 5) -> str:
    '''
    Chama o endpoint correto do Ollama (/api/chat ou /api/generate) conforme o modelo.
    '''
    # Lista de modelos conhecidos que suportam /api/chat
    chat_models = [
        "llama2", "llama3", "qwen2.5:7b-instruct", "qwen2.5:14b", "mistral", "gemma", "phi3", "dolphin", "codellama"
    ]
    use_chat = any(m in model for m in chat_models)
    for attempt in range(1, max_retries+1):
        try:
            if use_chat:
                url = f"{OLLAMA_HOST}/api/chat"
                payload = {
                    "model": model,
                    "messages": messages,
                    "options": {
                        "temperature": temperature,
                    },
                    "stream": False
                }
                r = requests.post(url, json=payload, timeout=120)
                if r.status_code == 404:
                    # Fallback para /api/generate se /api/chat não existir
                    use_chat = False
                else:
                    r.raise_for_status()
                    data = r.json()
                    return data.get("message", {}).get("content", "")
            if not use_chat:
                # Para modelos pequenos ou fallback, usa /api/generate
                url = f"{OLLAMA_HOST}/api/generate"
                prompt = "\n\n".join([m["content"] for m in messages])
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "options": {
                        "temperature": temperature,
                    },
                    "stream": False
                }
                r = requests.post(url, json=payload, timeout=120)
                r.raise_for_status()
                data = r.json()
                return data.get("response", "")
        except Exception as e:
            if attempt == max_retries:
                raise
            time.sleep(1.5 * attempt)
    return ""

MAP_SYS = "Você é um assistente que resume com precisão técnica e em PT-BR."
REDUCE_SYS = "Você é um assistente que consolida resumos em PT-BR de forma estruturada e objetiva."

def map_reduce_summarize(
    chunks: List[str],
    model: str,
    language: str = "pt-BR",
    map_prompt: str = "Resuma objetivamente (bullet points, PT-BR) o trecho a seguir, preservando números, entidades e termos técnicos:",
    reduce_prompt: str = "Junte os resumos em um único resumo executivo em PT-BR, com seções curtas e claras. Foque em objetivos, métricas, decisões e riscos."
) -> str:
    # 1. Sumarização MAP: cada chunk é resumido individualmente
    partials = []
    for i, chunk in enumerate(chunks):
        logging.info(f"[MAP] Resumindo chunk {i+1}/{len(chunks)}")
        user = f"{map_prompt}\n\n---\n{chunk}\n---"
        try:
            txt = _chat(model, [
                {"role": "system", "content": MAP_SYS},
                {"role": "user", "content": user}
            ])
            logging.info(f"[MAP] Chunk {i+1} resumido.")
        except Exception as e:
            logging.error(f"Erro ao resumir chunk {i+1}: {e}")
            raise
        partials.append(f"## Parte {i+1}\n{txt.strip()}")

    # 2. Sumarização REDUCE: juntar todos os resumos em um só
    reduce_input = "\n\n".join(partials)
    logging.info("[REDUCE] Consolidando todos os resumos em um único texto.")
    try:
        final = _chat(model, [
            {"role": "system", "content": REDUCE_SYS},
            {"role": "user", "content": f"{reduce_prompt}\n\n{reduce_input}"}
        ])
        logging.info("[REDUCE] Resumo final gerado.")
    except Exception as e:
        logging.error(f"Erro ao consolidar resumo final: {e}")
        raise

    # 3. Retornar o resumo final
    return final.strip()
