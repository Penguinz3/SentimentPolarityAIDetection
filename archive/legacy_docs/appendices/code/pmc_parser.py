import re
import xml.etree.ElementTree as ET
from pathlib import Path


WHITESPACE_RE = re.compile(r"\s+")


def _text_content(elem):
    if elem is None:
        return ""
    text = "".join(elem.itertext())
    return WHITESPACE_RE.sub(" ", text).strip()


def parse_pmc_xml(xml_path):
    """
    Minimal PMC/JATS parser.

    Returns:
      meta: dict with doc_id, journal, year, title
      sections: list of dicts with section (name) and text
    """
    xml_path = Path(xml_path)
    root = ET.parse(xml_path).getroot()

    # Metadata
    journal = ""
    journal_title = root.find(".//journal-title")
    if journal_title is not None:
        journal = _text_content(journal_title)

    article_title = ""
    title_elem = root.find(".//article-title")
    if title_elem is not None:
        article_title = _text_content(title_elem)

    doc_id = xml_path.stem
    pmcid_elem = root.find(".//article-id[@pub-id-type='pmcid']")
    if pmcid_elem is not None and _text_content(pmcid_elem):
        doc_id = _text_content(pmcid_elem)

    year = ""
    # Prefer epub/ppub, fall back to first year found.
    for xpath in [
        ".//pub-date[@pub-type='epub']/year",
        ".//pub-date[@pub-type='ppub']/year",
        ".//pub-date/year",
    ]:
        year_elem = root.find(xpath)
        if year_elem is not None and _text_content(year_elem):
            year = _text_content(year_elem)
            break

    meta = {"doc_id": doc_id, "journal": journal, "year": year, "title": article_title}

    # Sections: pull body <sec> blocks and gather <p> text under each.
    sections = []
    body = root.find(".//body")
    if body is None:
        return meta, sections

    for sec in body.findall(".//sec"):
        sec_title = _text_content(sec.find("./title")) or "unknown"
        paras = []
        for p in sec.findall(".//p"):
            txt = _text_content(p)
            if txt:
                paras.append(txt)
        sec_text = " ".join(paras).strip()
        if sec_text:
            sections.append({"section": sec_title.lower(), "text": sec_text})

    return meta, sections

