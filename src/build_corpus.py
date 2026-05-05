from __future__ import annotations

import argparse
import csv
import re
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path


WHITESPACE_RE = re.compile(r"\s+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def text_content(element: ET.Element | None) -> str:
    if element is None:
        return ""
    text = "".join(element.itertext())
    return WHITESPACE_RE.sub(" ", text).strip()


def parse_pmc_xml(xml_path: str | Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    xml_path = Path(xml_path)
    root = ET.parse(xml_path).getroot()

    journal = text_content(root.find(".//journal-title"))
    article_title = text_content(root.find(".//article-title"))

    doc_id = xml_path.stem
    pmcid_element = root.find(".//article-id[@pub-id-type='pmcid']")
    if pmcid_element is not None and text_content(pmcid_element):
        doc_id = text_content(pmcid_element)

    year = ""
    for xpath in [
        ".//pub-date[@pub-type='epub']/year",
        ".//pub-date[@pub-type='ppub']/year",
        ".//pub-date/year",
    ]:
        year_element = root.find(xpath)
        if year_element is not None and text_content(year_element):
            year = text_content(year_element)
            break

    meta = {"doc_id": doc_id, "journal": journal, "year": year, "title": article_title}

    body = root.find(".//body")
    if body is None:
        return meta, []

    sections = []
    for section in body.findall(".//sec"):
        section_title = text_content(section.find("./title")) or "unknown"
        paragraphs = [text_content(p) for p in section.findall(".//p")]
        section_text = " ".join(p for p in paragraphs if p).strip()
        if section_text:
            sections.append({"section": section_title.lower(), "text": section_text})
    return meta, sections


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [sentence.strip() for sentence in SENTENCE_SPLIT_RE.split(text) if sentence.strip()]


def chunk_sentences(sentences: list[str], sentences_per_chunk: int):
    for index in range(0, len(sentences), sentences_per_chunk):
        yield index, sentences[index : index + sentences_per_chunk]


def build_corpus(pmc_dir: Path, output_path: Path, sentences_per_chunk: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    namespace = uuid.UUID("12345678-1234-5678-1234-567812345678")
    fieldnames = [
        "chunk_id",
        "doc_id",
        "source",
        "year",
        "journal",
        "section",
        "chunk_index",
        "start_sentence",
        "n_sentences",
        "word_count",
        "text",
        "raw_doc_path",
    ]

    xml_files = sorted(pmc_dir.glob("*.xml"))
    if not xml_files:
        raise SystemExit(f"No PMC XML files found under: {pmc_dir}")

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for xml_path in xml_files:
            meta, sections = parse_pmc_xml(xml_path)
            section_counts: dict[str, int] = {}

            for section_payload in sections:
                section = section_payload["section"]
                sentences = split_sentences(section_payload["text"])
                if not sentences:
                    continue

                chunk_index = section_counts.get(section, 0)
                for start_sentence, chunk_text_sentences in chunk_sentences(
                    sentences, sentences_per_chunk
                ):
                    chunk_text = " ".join(chunk_text_sentences)
                    chunk_id = str(uuid.uuid5(namespace, f"{meta['doc_id']}:{section}:{chunk_index}"))
                    writer.writerow(
                        {
                            "chunk_id": chunk_id,
                            "doc_id": meta["doc_id"],
                            "source": "PMC",
                            "year": meta["year"],
                            "journal": meta["journal"],
                            "section": section,
                            "chunk_index": chunk_index,
                            "start_sentence": start_sentence,
                            "n_sentences": len(chunk_text_sentences),
                            "word_count": len(chunk_text.split()),
                            "text": chunk_text,
                            "raw_doc_path": str(xml_path),
                        }
                    )
                    chunk_index += 1

                section_counts[section] = chunk_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a chunked PMC corpus from JATS XML files.")
    parser.add_argument("--pmc-dir", default="data/raw/PMC", help="Directory containing PMC XML files.")
    parser.add_argument("--output", default="outputs/corpus_chunks.csv", help="Output CSV path.")
    parser.add_argument("--sentences-per-chunk", type=int, default=8, help="Sentences per chunk.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_corpus(Path(args.pmc_dir), Path(args.output), args.sentences_per_chunk)
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()

