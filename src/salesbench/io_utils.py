from __future__ import annotations

import csv
import json
from pathlib import Path


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def read_csv(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_records(path: Path) -> list[dict[str, object]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl(path)
    if suffix == ".csv":
        return read_csv(path)
    if suffix == ".json":
        payload = read_json(path)
        if isinstance(payload, list):
            return payload
        raise ValueError(f"{path} 不是记录数组格式的 JSON 文件")
    raise ValueError(f"暂不支持读取 {suffix} 格式：{path}")
