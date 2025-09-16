from typing import List, Tuple

def chunk_text(text: str, chunk_size: int = 450000, overlap: int = 400) -> List[Tuple[str, int]]:
    if chunk_size <= 0:
        chunk_size = 450000
    if overlap < 0:
        overlap = 400
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + chunk_size, n)
        chunk = text[i:end]
        chunks.append((chunk, i))
        if end == n:
            break
        i = end - overlap
        if i < 0:
            i = 0
    return chunks
