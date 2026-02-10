import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

from docx import Document

EXPORT_DIR = "exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

MAX_VERSIONS = 5

DISCLAIMER = (
    "LEGAL DISCLAIMER:\n"
    "This document is a non-binding employment contract draft.\n"
    "It is for informational purposes only and does not constitute legal advice.\n"
)


def _ensure_disclaimer(text: str) -> str:
    if not text:
        return text
    if "LEGAL DISCLAIMER:" in text:
        return text
    return text.rstrip() + "\n\n" + DISCLAIMER


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class ContractVersion:
    text: str
    created_at: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContractStore:
    current: Optional[str] = None
    versions: List[ContractVersion] = field(default_factory=list)

    def save_version(self, meta: Optional[Dict[str, Any]] = None) -> None:
        """
        Save current into history (up to MAX_VERSIONS), most recent first.
        """
        if self.current:
            self.versions.insert(0, ContractVersion(
                text=self.current,
                created_at=_now_iso(),
                meta=meta or {}
            ))
            self.versions = self.versions[:MAX_VERSIONS]

    def get_previous(self, index: int = 0) -> Optional[str]:
        if 0 <= index < len(self.versions):
            return self.versions[index].text
        return None

    def list_versions(self) -> List[Dict[str, Any]]:
        return [{"index": i, "created_at": v.created_at, "meta": v.meta} for i, v in enumerate(self.versions)]


def generate_contract(store: ContractStore, data: Dict[str, str]) -> str:
    contract = f"""
NON-BINDING EMPLOYMENT CONTRACT DRAFT

This Employment Contract is made on: {data.get('date', '')}

BETWEEN:
Employer: {data.get('employer', '')}
Location: {data.get('location', '')}

AND
Employee: {data.get('employee', '')}
Job Title: {data.get('job_title', '')}

TERM:
Contract type: {data.get('contract_type', '')}
Start Date: {data.get('start_date', '')}
End Date: {data.get('end_date', '')}

COMPENSATION:
Salary: {data.get('salary', '')} JOD

WORK SCHEDULE:
Working Hours: {data.get('hours', '')} hours/day
Working Days: {data.get('days', '')} days/week

TERMINATION:
{data.get('termination', '')}

{DISCLAIMER}
""".strip()

    # Save old current first
    store.save_version(meta={"action": "generate"})
    store.current = _ensure_disclaimer(contract)
    return store.current


SECTION_PATTERNS = {
    # Keep patterns stable (single-line sections)
    "salary": r"(Salary:\s*)(.*)",
    "location": r"(Location:\s*)(.*)",
    "job_title": r"(Job Title:\s*)(.*)",
    # Multi-line termination: keep boundary before disclaimer
    "termination": r"(TERMINATION:\n)([\s\S]*?)(\n\nLEGAL DISCLAIMER:)",
}


def update_section(store: ContractStore, section: str, new_value: str) -> str:
    if not store.current:
        return "❌ No contract exists. Generate a contract first."

    section = (section or "").strip().lower()
    if section not in SECTION_PATTERNS:
        return "❌ Unknown section. Use: salary, location, job_title, termination"

    if new_value is None:
        new_value = ""
    new_value = new_value.strip()

    # Save old before editing
    store.save_version(meta={"action": "update", "section": section})

    text = store.current

    if section == "termination":
        m = re.search(SECTION_PATTERNS["termination"], text)
        if not m:
            # fallback: add termination before disclaimer safely
            if "LEGAL DISCLAIMER:" in text:
                head, tail = text.split("LEGAL DISCLAIMER:", 1)
                text = head.rstrip() + "\n\nTERMINATION:\n" + new_value + "\n\nLEGAL DISCLAIMER:" + tail
            else:
                text = text.rstrip() + "\n\nTERMINATION:\n" + new_value + "\n\n" + DISCLAIMER
        else:
            text = re.sub(
                SECTION_PATTERNS["termination"],
                r"\1" + new_value + r"\3",
                text
            )
    else:
        # Replace only the first match for safety
        text = re.sub(
            SECTION_PATTERNS[section],
            lambda mm: mm.group(1) + new_value,
            text,
            count=1
        )

    store.current = _ensure_disclaimer(text)
    return store.current


def export_to_word(contract_text: str) -> str:
    if not contract_text:
        return "❌ No contract to export."

    contract_text = _ensure_disclaimer(contract_text)

    doc = Document()
    # Basic formatting: title bigger, rest normal (still simple)
    lines = contract_text.split("\n")
    if lines:
        doc.add_heading(lines[0].strip(), level=1)
        for line in lines[1:]:
            doc.add_paragraph(line)

    path = os.path.join(EXPORT_DIR, f"contract_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx")
    doc.save(path)
    return path
