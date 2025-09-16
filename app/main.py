import os
import time
import traceback
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
import uuid
import shutil
from typing import Optional
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import PlainTextResponse, FileResponse
from pydantic import BaseModel

from app.ocr_extract import ensure_searchable_pdf, extract_text_from_pdf
from app.chunking import chunk_text
from app.summarize import map_reduce_summarize
from app.vector_index import VectorIndex

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "./data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen2.5:7b-instruct")

app = FastAPI(title="Offline PDF Summarizer (No Docker)",
              version="1.0.0",
              description="Upload PDF -> OCR/extract -> index (Chroma local) -> map-reduce summarize via Ollama -> return .txt")

class SummarizeResponse(BaseModel):
    job_id: str
    model: str
    output_path: str
    tokens_estimate: Optional[int] = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/summarize", response_class=PlainTextResponse)
def summarize_pdf(
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL),
    chunk_chars: int = Form(450000),
    overlap: int = Form(400),
    language: str = Form("pt-BR"),
    map_prompt: str = Form("Resuma objetivamente (bullet points, PT-BR) o trecho a seguir, preservando números, entidades e termos técnicos:"),
    reduce_prompt: str = Form("Junte os resumos em um único resumo executivo em PT-BR, com seções curtas e claras. Foque em objetivos, métricas, decisões e riscos.")
):
    # 1. Gerar um ID único para o job
    job_id = str(uuid.uuid4())
    start_total = time.time()
    logging.info(f"[START] Job {job_id} - Model: {model}")

    # 2. Salvar o arquivo recebido
    raw_path = os.path.join(UPLOAD_DIR, f"{job_id}.pdf")
    with open(raw_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    logging.info(f"PDF salvo em {raw_path}")

    # 3. Tornar o PDF pesquisável via OCR
    searchable_path = os.path.join(UPLOAD_DIR, f"{job_id}.searchable.pdf")
    try:
        start_ocr = time.time()
        ensure_searchable_pdf(raw_path, searchable_path)
        logging.info(f"OCR concluído em {time.time() - start_ocr:.2f}s")
    except Exception as e:
        logging.error(f"OCR failed: {e}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")

    # 4. Extrair texto do PDF
    try:
        start_extract = time.time()
        text = extract_text_from_pdf(searchable_path)
        logging.info(f"Extração de texto concluída em {time.time() - start_extract:.2f}s")
    except Exception as e:
        logging.error(f"Extraction failed: {e}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")

    # 5. Verificar se o texto foi extraído
    if not text or len(text.strip()) == 0:
        logging.warning("Nenhum texto extraído do PDF. Verifique a qualidade do arquivo.")
        raise HTTPException(status_code=400, detail="Nenhum texto extraído do PDF. Verifique a qualidade do arquivo.")

    # 6. Dividir texto em chunks
    start_chunk = time.time()
    chunks = chunk_text(text, chunk_size=int(chunk_chars), overlap=int(overlap))
    logging.info(f"Chunking concluído ({len(chunks)} chunks) em {time.time() - start_chunk:.2f}s")

    # 7. Indexar chunks no vetor
    index = VectorIndex()
    index.reset()  # simple behavior; swap for per-job collection if needed
    index.add_documents([c for c,_ in chunks], metadatas=[{"job_id": job_id, "seq": i} for i,_ in enumerate(chunks)])
    logging.info("Indexação concluída.")

    # 8. Sumarizar os chunks usando o modelo
    try:
        start_summarize = time.time()
        final_summary = map_reduce_summarize(
            chunks=[c for c,_ in chunks],
            model=model,
            language=language,
            map_prompt=map_prompt,
            reduce_prompt=reduce_prompt
        )
        logging.info(f"Sumarização concluída em {time.time() - start_summarize:.2f}s")
    except Exception as e:
        logging.error(f"Summarization failed: {e}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Summarization failed: {e}")

    # 9. Salvar o resumo em arquivo
    out_path = os.path.join(OUTPUT_DIR, f"{job_id}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(final_summary.strip())
    logging.info(f"Resumo salvo em {out_path}")
    logging.info(f"[END] Job {job_id} - Tempo total: {time.time() - start_total:.2f}s")

    # 10. Retornar o resumo como resposta
    return PlainTextResponse(final_summary, media_type="text/plain; charset=utf-8")

@app.get("/download/{job_id}")
def download(job_id: str):
    txt_path = os.path.join(OUTPUT_DIR, f"{job_id}.txt")
    if not os.path.exists(txt_path):
        raise HTTPException(status_code=404, detail="Resultado não encontrado.")
    return FileResponse(txt_path, filename=f"{job_id}.txt", media_type="text/plain")
