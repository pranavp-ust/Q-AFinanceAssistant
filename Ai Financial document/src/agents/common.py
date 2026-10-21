"""Helpers shared by agents."""
import re

ADVICE_PATTERN = re.compile(
    r"\b(should i (buy|sell|invest|hold)|is it (a )?good (investment|time to buy)|"
    r"(buy|sell|hold) (the )?(stock|shares)|investment advice|price target|recommend.*(stock|shares|invest))\b",
    re.IGNORECASE,
)

STOPWORDS = set("""a an the of in on for to and or is was were what which who how much many did does do
be by with from as at it its this that these those their there about fy company report annual please tell me
give show""".split())


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def body(chunk: dict) -> str:
    """Chunk text without the [Company | FY | doc] header added at ingestion."""
    return normalize(re.sub(r"^\[[^\]]*\]\s*", "", chunk["text"]))


def format_context(chunks: list[dict], limit: int = 8) -> str:
    parts = []
    for i, c in enumerate(chunks[:limit], start=1):
        label = f"{c.get('company') or 'General'} FY{c.get('fiscal_year') or '-'} | {c.get('source')} p.{c.get('page')}"
        parts.append(f"[{i}] ({label})\n{body(c)}")
    return "\n\n".join(parts)


def citations_for(chunks: list[dict], limit: int = 8) -> list[dict]:
    return [{"n": i, "source": c.get("source"), "page": c.get("page"),
             "company": c.get("company"), "fiscal_year": c.get("fiscal_year"),
             "distance": c.get("distance")}
            for i, c in enumerate(chunks[:limit], start=1)]


def sentences(text: str) -> list[str]:
    """Prose sentences only - table rows (many bare numbers) are skipped."""
    out = []
    for s in re.split(r"(?<=[.!?])\s+(?=[A-Z•&])|\s(?=•)", normalize(text)):
        s = s.strip()
        if len(s) > 20 and len(re.findall(r"\b\d[\d,]*\.\d\b(?! million)", s)) <= 3:
            out.append(s)
    return out


def keywords(text: str) -> set[str]:
    words = (w.strip("'-") for w in re.findall(r"[a-z][a-z\-']+", text.lower()))
    return {w[:-1] if len(w) > 4 and w.endswith("s") else w
            for w in words if w not in STOPWORDS and len(w) > 2}


def numbers_in(text: str) -> set[str]:
    """Numbers like 1,234.5 / 12.3 / 0.45 (years excluded), normalised without commas."""
    out = set()
    for n in re.findall(r"\d[\d,]*\.?\d*", text):
        n = n.replace(",", "").rstrip(".")
        if not n or re.fullmatch(r"(19|20)\d{2}", n) or (len(n) == 1):
            continue
        out.add(n)
    return out


def money(v) -> str:
    return "n/a" if v is None else f"${v:,.1f}M"
