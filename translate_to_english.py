"""Translate cs/sk prose documents across all submodules to English using local
Helsinki-NLP (OPUS-MT / MarianMT) models.

embedder_config.toml's [translation] section names "Helsinki-NLP" as the engine,
but rag_pipeline.RAGPipeline actually translates via the DeepL API. This script
runs the real thing. It instantiates the same RAGPipeline class main.py drives
for embedding — reusing its dataset_dir/submodule layout and the exact
get_file_metadata/extract_file_text/is_prose_document helpers it relies on for
text extraction and metadata — but performs translation locally via MarianMT
instead of calling out to DeepL.

For each project (submodule) under --dataset-dir, output goes to
<output-dir>/<project>/:
  - every translated file as "<stem>.md" + "<stem>_metadata.json"
  - a single "non-prose.json" listing every file in the project that was *not*
    translated, with a reason (not prose, wrong language, empty, extraction/
    translation failure, ...)

Usage:
    python translate_to_english.py [--dataset-dir submodules] [--output-dir translations] [--overwrite]
"""

import argparse
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "models"))
MODELS_DIR.mkdir(exist_ok=True)
os.environ.setdefault("HF_HOME", str(MODELS_DIR.resolve()))

import torch
from transformers import MarianMTModel, MarianTokenizer

from rag_pipeline import (
    RAGPipeline, extract_file_text, get_file_metadata, load_model_configs, load_translation_config,
)
from rag_pipeline.pipeline import detect_language, is_prose_document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Same file-discovery set main.py's process_docs() scans, so this walks the
# identical corpus that gets fed to the embedding pipeline.
FILE_TYPES = {
    ".md", ".pdf", ".yaml", ".html", ".txt", ".rb", ".java", ".json",
    ".go", ".php", ".js", ".tx", ".cs", ".cshtml", ".csv", ".groovy", ".bat", ".sh",
}

# Dedicated OPUS-MT single-pair models for the two source languages in scope.
LANG_MODEL_IDS = {
    "cs": "Helsinki-NLP/opus-mt-cs-en",
    "sk": "Helsinki-NLP/opus-mt-sk-en",
}

# Human-readable abbreviations for the metadata "language" field, e.g. "sk -> eng".
LANG_DISPLAY = {"cs": "cz", "sk": "sk"}
TARGET_LANG_DISPLAY = "eng"

MAX_WORDS_PER_CHUNK = 180  # keeps subword-tokenized chunks safely under the 512-token model limit
TRANSLATE_BATCH_SIZE = 8

_model_cache: dict[str, tuple[MarianTokenizer, MarianMTModel]] = {}


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_translator(lang: str, device: str) -> tuple[MarianTokenizer, MarianMTModel]:
    model_id = LANG_MODEL_IDS[lang]
    if model_id not in _model_cache:
        log.info("Loading translation model %s ...", model_id)
        tokenizer = MarianTokenizer.from_pretrained(model_id)
        model = MarianMTModel.from_pretrained(model_id).to(device).eval()
        _model_cache[model_id] = (tokenizer, model)
    return _model_cache[model_id]


def _sentence_chunks(paragraph: str, max_words: int) -> list[str]:
    """Pack a paragraph's sentences into word-bounded chunks without splitting a
    sentence unless it alone exceeds max_words."""
    sentences = re.split(r"(?<=[.!?])\s+", paragraph.strip())
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for sentence in sentences:
        words = len(sentence.split())
        if current and current_words + words > max_words:
            chunks.append(" ".join(current))
            current, current_words = [], 0
        current.append(sentence)
        current_words += words
    if current:
        chunks.append(" ".join(current))

    final: list[str] = []
    for chunk in chunks:
        words = chunk.split()
        if len(words) <= max_words:
            final.append(chunk)
        else:
            for i in range(0, len(words), max_words):
                final.append(" ".join(words[i : i + max_words]))
    return final


_CODE_FENCE_RE = re.compile(r"^\s*```")


def _translate_chunks(
    chunks: list[str], tokenizer: MarianTokenizer, model: MarianMTModel, device: str, batch_size: int,
) -> list[str]:
    outputs: list[str] = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            generated = model.generate(**encoded, max_length=512)
        outputs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return outputs


def translate_text(text: str, lang: str, device: str) -> str:
    """Translate `text` line by line, so markdown structure (headings, list items,
    blank lines) survives, and fenced code blocks are passed through untranslated
    rather than corrupted by machine translation."""
    tokenizer, model = _get_translator(lang, device)

    lines = text.split("\n")
    line_chunk_lists: list[list[str] | None] = []
    flat_chunks: list[str] = []
    in_code_block = False
    for line in lines:
        if _CODE_FENCE_RE.match(line):
            in_code_block = not in_code_block
            line_chunk_lists.append(None)
            continue
        if in_code_block or not line.strip():
            line_chunk_lists.append(None)
            continue
        chunks = _sentence_chunks(line, MAX_WORDS_PER_CHUNK)
        line_chunk_lists.append(chunks)
        flat_chunks.extend(chunks)

    if not flat_chunks:
        return ""

    translated_flat = _translate_chunks(flat_chunks, tokenizer, model, device, TRANSLATE_BATCH_SIZE)

    translated_iter = iter(translated_flat)
    out_lines = [
        line if chunk_list is None else " ".join(next(translated_iter) for _ in chunk_list)
        for line, chunk_list in zip(lines, line_chunk_lists)
    ]
    return "\n".join(out_lines)


def _existing_metadata_matches(json_path: Path, source_relative_path: str) -> bool:
    """Whether an existing "<stem>_metadata.json" sidecar was written for this exact
    source file (vs. a different file that merely shares the same stem)."""
    if not json_path.exists():
        return False
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return data.get("source_relative_path") == source_relative_path


def _resolve_output_paths(
    project_output_dir: Path, stem: str, source_relative_path: str,
) -> tuple[Path, Path, bool]:
    """Pick the "<stem>.md" / "<stem>_metadata.json" pair for one source file inside a
    flat per-project output folder.

    Multiple source files across different subdirectories of a project can share a
    stem (e.g. two READMEs). Rather than silently overwriting one translation with
    another, an occupied stem belonging to a *different* source file is disambiguated
    with a "__2", "__3", ... suffix. Returns (md_path, json_path, already_translated).
    """
    candidate = stem
    n = 1
    while True:
        md_path = project_output_dir / f"{candidate}.md"
        json_path = project_output_dir / f"{candidate}_metadata.json"
        if not md_path.exists() and not json_path.exists():
            return md_path, json_path, False
        if _existing_metadata_matches(json_path, source_relative_path):
            return md_path, json_path, True
        n += 1
        candidate = f"{stem}__{n}"


def process_file(
    filepath: Path, dataset_dir: Path, project: str, project_output_dir: Path,
    device: str, supported_langs: set[str], overwrite: bool,
) -> str:
    """Translate one file if eligible. Returns a status string for logging/summary;
    every non-"translated" status is what process_project() records into non-prose.json."""
    if not is_prose_document(filepath):
        return "not_prose"

    source_relative_path = str(filepath.relative_to(dataset_dir))
    md_path, json_path, already_translated = _resolve_output_paths(
        project_output_dir, filepath.stem, source_relative_path
    )
    if already_translated and not overwrite:
        return "skipped_exists"

    try:
        text = extract_file_text(filepath)
    except Exception as e:
        log.warning("Could not extract text from %s: %s", filepath, e)
        return "failed_extract"

    if not text.strip():
        return "empty_content"

    source_lang = detect_language(text)
    if source_lang not in supported_langs:
        return f"unsupported_language:{source_lang}"

    try:
        translated = translate_text(text, source_lang, device)
    except Exception as e:
        log.warning("Translation failed for %s: %s", filepath, e)
        return "failed_translate"

    if not translated.strip():
        return "empty_translation"

    md_path.write_text(translated, encoding="utf-8")

    file_meta = get_file_metadata(filepath)
    metadata = {
        "source_path": str(filepath.resolve()),
        "source_relative_path": source_relative_path,
        "project": project,
        "file_type": filepath.suffix.lower(),
        "source_language": source_lang,
        "target_language": "en",
        "language": f"{LANG_DISPLAY.get(source_lang, source_lang)} -> {TARGET_LANG_DISPLAY}",
        "translation_engine": "Helsinki-NLP",
        "translation_model_id": LANG_MODEL_IDS[source_lang],
        "translated_at": datetime.now().isoformat(),
        "source_char_count": len(text),
        "translated_char_count": len(translated),
        "authors": file_meta["authors"],
        "created_at": file_meta["created_at"],
        "modified_at": file_meta["modified_at"],
    }
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return "translated"


def process_project(
    project: str, dataset_dir: Path, output_dir: Path, device: str, supported_langs: set[str], overwrite: bool,
) -> dict[str, int]:
    project_source_dir = dataset_dir / project
    project_output_dir = output_dir / project
    project_output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        f for f in project_source_dir.rglob("*")
        if f.is_file() and (f.suffix.lower() in FILE_TYPES or f.stem == "License")
    )
    log.info("[%s] %d file(s) to check", project, len(files))

    counts: dict[str, int] = {}
    non_prose_entries = []
    for filepath in files:
        status = process_file(filepath, dataset_dir, project, project_output_dir, device, supported_langs, overwrite)
        counts[status] = counts.get(status, 0) + 1
        log.info("[%s] %s -> %s", project, filepath.relative_to(project_source_dir), status)
        if status not in ("translated", "skipped_exists"):
            non_prose_entries.append({
                "source_relative_path": str(filepath.relative_to(dataset_dir)),
                "reason": status,
            })

    non_prose_path = project_output_dir / "non-prose.json"
    non_prose_path.write_text(json.dumps(non_prose_entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log.info("[%s] Summary: %s", project, counts)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="submodules", help="Root directory of project submodules to scan")
    parser.add_argument("--output-dir", default="translations", help="Where translated .md/.json pairs are written")
    parser.add_argument("--overwrite", action="store_true", help="Re-translate files that already have output")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    device = _device()
    log.info("Using device: %s", device)

    model_configs = load_model_configs()
    translation_config = load_translation_config()
    # Reuses RAGPipeline purely for its dataset_dir/submodule conventions and the
    # get_file_metadata/extract_file_text/is_prose_document helpers it relies on for
    # embedding — no embedder is ever loaded, and Qdrant/Pgvector are never written to
    # (RAGPipeline.__init__ does a harmless version-check ping against Qdrant, logged
    # as a warning if it's not running — nothing here depends on that succeeding).
    rag = RAGPipeline(model_configs, translation_config, dataset_dir=Path(args.dataset_dir))

    configured_langs = set(translation_config.get("languages", LANG_MODEL_IDS.keys()))
    supported_langs = configured_langs & LANG_MODEL_IDS.keys()
    unsupported = configured_langs - LANG_MODEL_IDS.keys()
    if unsupported:
        log.warning("No Helsinki-NLP model configured for language(s) %s — skipping those", unsupported)

    projects = sorted(d.name for d in rag.dataset_dir.iterdir() if d.is_dir())
    log.info("Found %d project(s) under %s", len(projects), rag.dataset_dir)

    totals: dict[str, int] = {}
    for project in projects:
        counts = process_project(project, rag.dataset_dir, output_dir, device, supported_langs, args.overwrite)
        for status, n in counts.items():
            totals[status] = totals.get(status, 0) + n

    log.info("Done. Grand total: %s", totals)


if __name__ == "__main__":
    main()
