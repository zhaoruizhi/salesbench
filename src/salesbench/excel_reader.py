from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

from .utils import clean_text


NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pkgrel": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def _column_index(cell_ref: str) -> int:
    letters = "".join(char for char in cell_ref if char.isalpha())
    index = 0
    for char in letters:
        index = index * 26 + (ord(char.upper()) - 64)
    return index - 1


def _load_shared_strings(archive: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    shared_strings: list[str] = []
    for item in root.findall("main:si", NS):
        texts = [node.text or "" for node in item.findall(".//main:t", NS)]
        shared_strings.append("".join(texts))
    return shared_strings


def _workbook_sheets(archive: ZipFile) -> list[tuple[str, str]]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    rel_map = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in relationships.findall("pkgrel:Relationship", NS)
    }

    sheets: list[tuple[str, str]] = []
    for sheet in workbook.findall("main:sheets/main:sheet", NS):
        relationship_id = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        target = rel_map[relationship_id]
        if not target.startswith("worksheets/"):
            continue
        sheets.append((sheet.attrib["name"], f"xl/{target}"))
    return sheets


def list_sheet_names(workbook_path: Path) -> list[str]:
    with ZipFile(workbook_path) as archive:
        return [name for name, _ in _workbook_sheets(archive)]


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    value_type = cell.attrib.get("t")
    if value_type == "s":
        value_node = cell.find("main:v", NS)
        if value_node is None or value_node.text is None:
            return ""
        return shared_strings[int(value_node.text)]

    if value_type == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(".//main:t", NS))

    value_node = cell.find("main:v", NS)
    if value_node is None or value_node.text is None:
        return ""
    return value_node.text


def _row_values(row: ET.Element, shared_strings: list[str]) -> list[str]:
    values: defaultdict[int, str] = defaultdict(str)
    for cell in row.findall("main:c", NS):
        cell_ref = cell.attrib.get("r", "")
        if not cell_ref:
            continue
        values[_column_index(cell_ref)] = _cell_value(cell, shared_strings)

    if not values:
        return []

    max_index = max(values)
    return [clean_text(values[index]) for index in range(max_index + 1)]


def _prepare_headers(raw_headers: list[str]) -> list[str]:
    counters: dict[str, int] = {}
    prepared: list[str] = []
    for index, header in enumerate(raw_headers, start=1):
        candidate = clean_text(header)
        if not candidate:
            candidate = f"unnamed_{index}"
        duplicate_count = counters.get(candidate, 0)
        counters[candidate] = duplicate_count + 1
        if duplicate_count:
            candidate = f"{candidate}__{duplicate_count + 1}"
        prepared.append(candidate)
    return prepared


def load_sheet_records(workbook_path: Path, sheet_name: str | None = None) -> list[dict[str, str]]:
    with ZipFile(workbook_path) as archive:
        shared_strings = _load_shared_strings(archive)
        sheets = _workbook_sheets(archive)
        if not sheets:
            return []

        target_sheet_name, target_path = sheets[0]
        if sheet_name is not None:
            for current_name, current_path in sheets:
                if current_name == sheet_name:
                    target_sheet_name = current_name
                    target_path = current_path
                    break
            else:
                available = ", ".join(name for name, _ in sheets)
                raise ValueError(f"未找到 sheet `{sheet_name}`，可选值：{available}")

        root = ET.fromstring(archive.read(target_path))
        rows = root.findall("main:sheetData/main:row", NS)
        if not rows:
            return []

        headers = _prepare_headers(_row_values(rows[0], shared_strings))
        records: list[dict[str, str]] = []
        for row in rows[1:]:
            values = _row_values(row, shared_strings)
            record = {
                header: clean_text(values[index]) if index < len(values) else ""
                for index, header in enumerate(headers)
            }
            if any(value != "" for value in record.values()):
                records.append(record)

        return records
