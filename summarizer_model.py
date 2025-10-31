import os
import math
from typing import List
from transformers import pipeline, AutoTokenizer
import torch

class SummarizerModel:
    def __init__(self,
                 model_name: str = "facebook/bart-large-cnn",
                 device: int = None,
                 chunk_overlap: int = 200,
                 chunk_size_chars: int = 3000):
        """
        model_name: HF model id
        device: None => auto detect (GPU if available)
        chunk_overlap: number of chars to overlap between chunks
        chunk_size_chars: approx chars per chunk before summarizing
        """
        self.model_name = model_name
        self.chunk_overlap = chunk_overlap
        self.chunk_size_chars = chunk_size_chars

        # Device selection
        if device is None:
            if torch.cuda.is_available():
                self.device = 0
            else:
                self.device = -1
        else:
            self.device = device

        # Load tokenizer to probe limits
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Build pipeline
        self.summarizer = pipeline(
            "summarization",
            model=self.model_name,
            tokenizer=self.tokenizer,
            device=self.device,
            truncation=True
        )

    def _chunk_text(self, text: str) -> List[str]:
        """Chunk long text into overlapping chunks using characters."""
        text = text.strip()
        if not text:
            return []
        max_chars = self.chunk_size_chars
        overlap = self.chunk_overlap
        if len(text) <= max_chars:
            return [text]

        chunks = []
        start = 0
        L = len(text)
        while start < L:
            end = start + max_chars
            chunk = text[start:end]
            
            chunks.append(chunk.strip())
            start = end - overlap
            if start < 0:
                start = 0
            if start >= L:
                break
        return chunks

    def summarize(self, text: str, summary_type: str = "short") -> str:
        """
        summary_type: short | medium | long
        Strategy:
          - Chunk long text
          - Summarize each chunk
          - If multiple chunk summaries -> summarize concatenated chunk summaries (hierarchical)
        """
        if not text or not text.strip():
            return ""

        if summary_type == "short":
            max_len, min_len = 60, 20
        elif summary_type == "medium":
            max_len, min_len = 150, 60
        else:  # long
            max_len, min_len = 300, 120

        chunks = self._chunk_text(text)
        if not chunks:
            return ""

        # Summarize each chunk
        chunk_summaries = []
        for chunk in chunks:
            try:
                out = self.summarizer(
                    chunk,
                    max_length=max_len,
                    min_length=min_len,
                    do_sample=False,
                    truncation=True
                )
                chunk_text = out[0]["summary_text"].strip()
            except Exception as e:
                # fallback: truncate and try again
                # ensure we don't crash entire pipeline
                chunk_text = chunk[:max_len * 4]
            chunk_summaries.append(chunk_text)

        # If multiple chunk summaries, summarize them again (hierarchical)
        if len(chunk_summaries) == 1:
            final_summary = chunk_summaries[0]
        else:
            concat = "\n\n".join(chunk_summaries)
            try:
                out = self.summarizer(
                    concat,
                    max_length=max_len,
                    min_length=min_len,
                    do_sample=False,
                    truncation=True
                )
                final_summary = out[0]["summary_text"].strip()
            except Exception:
                # fallback: join chunk summaries
                final_summary = " ".join(chunk_summaries)

        return final_summary