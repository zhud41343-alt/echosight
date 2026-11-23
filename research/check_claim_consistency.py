"""
P0-3 (V8.5) Automated Scientific & Documentation Claim Consistency Checker.
Fails with non-zero exit code if any formal document or UI component re-introduces:
- Deprecated/exaggerated terms: 'FG-EDFM', 'Frequency-Guided', '频域融合'
- Stale superseded ablation figures: '99.1 FPS', '16.8%', '16.3%'
- Outdated test counts: '40/40', '41/41'
- Overclaims: '全面领先', '全指标反超'
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent

FORMAL_FILES = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "TIANCHI_SUBMISSION_DRAFT.md",
    REPO_ROOT / "docs" / "COMPETITION_PPT_AND_VIDEO_SCRIPT.md",
    REPO_ROOT / "docs" / "REAL_BROWSER_TEST_REPORT.md",
    REPO_ROOT / "src" / "components" / "ValidationPanel.tsx",
    REPO_ROOT / "src" / "components" / "TechArchitecture.tsx",
    REPO_ROOT / "research" / "PAPER_DRAFT_EN_AND_PITCH.md",
    REPO_ROOT / "research" / "paper_ieee_template.tex",
    REPO_ROOT / "research" / "PATENT_DISCLOSURE_CN.md",
    REPO_ROOT / "research" / "outputs" / "ablation_report.md",
    REPO_ROOT / "research" / "outputs" / "MODEL_PROVENANCE.md",
]

FORBIDDEN_PATTERNS = [
    "FG-EDFM",
    "Frequency-Guided",
    "频域融合",
    "频域边缘",
    "99.1 FPS",
    "16.8%",
    "16.3%",
    "40/40",
    "41/41",
    "全面领先",
    "全指标反超",
    "WCAG AAA",
]


def main() -> int:
    violations: list[str] = []
    for path in FORMAL_FILES:
        if not path.exists():
            violations.append(f"MISSING FILE: {path}")
            continue
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            for pat in FORBIDDEN_PATTERNS:
                if pat in line:
                    violations.append(
                        f"{path.relative_to(REPO_ROOT)}:{line_no}: forbidden stale/exaggerated term '{pat}'"
                    )

    if violations:
        print("CLAIM CONSISTENCY AUDIT FAILED:")
        for v in violations:
            print("  -", v)
        return 1

    print(
        f"CLAIM CONSISTENCY AUDIT PASSED! Checked {len(FORMAL_FILES)} formal files against "
        f"{len(FORBIDDEN_PATTERNS)} forbidden stale/exaggerated patterns — 0 violations."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
