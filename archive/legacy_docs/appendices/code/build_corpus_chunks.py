import argparse
import csv
import re
import uuid
from pathlib import Path

from pmc_parser import parse_pmc_xml


SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text):
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]


def chunk_sentences(sentences, sentences_per_chunk):
    for i in range(0, len(sentences), sentences_per_chunk):
        yield i, sentences[i : i + sentences_per_chunk]


def main():
    ap = argparse.ArgumentParser(description="Build corpus_chunks.csv from PMC XML files.")
    ap.add_argument("--pmc-dir", default=str(Path("..") / "raw" / "PMC"), help="Directory containing PMC *.xml files.")
    ap.add_argument("--out", default="corpus_chunks.csv", help="Output CSV path.")
    ap.add_argument("--sentences-per-chunk", type=int, default=8, help="Sentences per chunk.")
    args = ap.parse_args()

    pmc_dir = Path(args.pmc_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ns = uuid.UUID("12345678-1234-5678-1234-567812345678")

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

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for xml_path in xml_files:
            meta, sections = parse_pmc_xml(xml_path)
            doc_id = meta.get("doc_id") or xml_path.stem
            year = meta.get("year") or ""
            journal = meta.get("journal") or ""

            # Track chunk_index per (doc, section) so ordering is stable.
            section_counts = {}
            for sec in sections:
                section = sec["section"]
                sentences = split_sentences(sec["text"])
                if not sentences:
                    continue

                idx = section_counts.get(section, 0)
                for start_sentence, chunk_sents in chunk_sentences(sentences, args.sentences_per_chunk):
                    chunk_text = " ".join(chunk_sents)
                    chunk_index = idx
                    idx += 1

                    chunk_id = str(uuid.uuid5(ns, f"{doc_id}:{section}:{chunk_index}"))
                    word_count = len(chunk_text.split())

                    writer.writerow(
                        {
                            "chunk_id": chunk_id,
                            "doc_id": doc_id,
                            "source": "PMC",
                            "year": year,
                            "journal": journal,
                            "section": section,
                            "chunk_index": chunk_index,
                            "start_sentence": start_sentence,
                            "n_sentences": len(chunk_sents),
                            "word_count": word_count,
                            "text": chunk_text,
                            "raw_doc_path": str(xml_path),
                        }
                    )

                section_counts[section] = idx

    print(f"Wrote: {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

