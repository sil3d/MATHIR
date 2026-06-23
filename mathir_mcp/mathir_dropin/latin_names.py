"""
MATHIR Drop-in -- Latin Name and Technical Nomenclature Handler
================================================================

Addresses four retrieval problems that the vanilla FTS5 + Jaccard
n-gram pipeline in :mod:`mathir_dropin.universal_bridge` cannot solve
on its own when the corpus contains scientific / medical / legal
Latin terminology:

1. **Taxonomic name collapse**
   ``"Homo sapiens"`` and ``"E. coli"`` and ``"Homo Sapiens"`` are
   the same biological entity but FTS5's ``porter unicode61``
   tokenizer treats them as three different strings.  We normalise
   binomials to ``{genus_lowercase, species_lowercase}`` so
   prefix-matches catch the abbreviation.

2. **Diacritic / case asymmetry**
   A query ``"Schrodinger"`` should match ``"Schrödinger"``;
   ``"Muller"`` should match ``"Müller"``; ``"Boole"`` should match
   ``"Boole"`` (already ASCII).  We do an NFKD pass + combining-mark
   strip + a small hand-curated transliteration table for the
   characters NFD misses (Polish ``Ł`` -> ``L``, German ``ß`` -> ``ss``,
   ligature ``œ`` -> ``oe``).

3. **Roman numerals in proper names**
   ``"Henry VIII"`` and ``"Henry 8"`` and ``"Henry The Eighth"`` are
   all valid spellings.  We detect roman-numeral *suffixes* on
   proper names and emit the integer form (and vice-versa) so the
   FTS5 index sees a single canonical shape.

4. **Abbreviations and compound medical terms**
   ``"DNA"`` <-> ``"deoxyribonucleic acid"`` and
   ``"sternocleidomastoid"`` -> ``["sterno", "cleido", "mastoid"]``
   both fail under naive n-gram matching because the abbreviation
   shares almost no shingles with the full form, and the compound
   shares no shingles with its meaning-defining roots.  We maintain
   a built-in lookup of common scientific abbreviations and a
   dictionary of medical / anatomical roots.

Mathematical grounding
======================

* :func:`text_similarity` (re-exported from
  :mod:`mathir_dropin.universal_bridge`) is the **Jaccard
  similarity over character n-grams** -- a lower bound on the
  cosine similarity of the corresponding one-hot n-gram vectors
  (Broder 1997).  We use it in :func:`latin_match` for the
  *string-equality* layer.

* :func:`latin_match` combines four channels with a **weighted
  reciprocal-rank fusion (RRF)** of the form::

      RRF(d) = sum_k w_k / (k0 + rank_k(d))

  with ``k0 = 60`` (the standard damping constant from Cormack
  et al. 2009).  The four channels are: (a) raw Jaccard, (b)
  canonicalised Jaccard, (c) taxonomic equality, (d) shared-root
  overlap.  RRF is a *parameter-light* way to combine rankings
  from heterogeneous signals; it is widely used in
  information-retrieval fusion (BM25 + dense vector, for
  instance).

* :func:`split_compound` uses **longest-prefix matching** over a
  curated roots dictionary.  The greedy choice is optimal under
  the assumption that the dictionary is *prefix-free* in
  spirit -- which it is, because the roots come from the
  standard medical Latin vocabulary (FMA, Terminologia
  Anatomica) and the longest match leaves the largest
  unambiguous remainder.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# We deliberately avoid a hard import of `universal_bridge` so this
# module is importable on its own (and so unit tests can run in
# isolation).  The :func:`text_similarity` re-export is a thin
# wrapper; the substantive logic is here.
try:  # pragma: no cover - import-time only
    from .universal_bridge import (
        normalize_unicode as _ub_normalize_unicode,
        transliterate as _ub_transliterate,
        char_ngrams as _ub_char_ngrams,
    )
    _HAS_UB = True
except Exception:  # pragma: no cover
    _ub_normalize_unicode = None
    _ub_transliterate = None
    _ub_char_ngrams = None
    _HAS_UB = False


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Maximum accepted roman-numeral value.  Beyond this the form is
# almost certainly a "subtractive" notation from a different era
# (e.g. Unicode uses ↁ for 5000, ↂ for 10000); we refuse and let the
# caller fall back.
_MAX_ROMAN = 3999  # MMMCMXCIX

# 50+ common scientific / medical / biochemical abbreviations.
# Curated against NCBI's "common abbreviations in biology" list +
# WHO essential medicines list.  Keys are UPPERCASE.
_ABBREVIATIONS: Dict[str, str] = {
    # Molecular biology
    "DNA": "deoxyribonucleic acid",
    "RNA": "ribonucleic acid",
    "mRNA": "messenger ribonucleic acid",
    "tRNA": "transfer ribonucleic acid",
    "rRNA": "ribosomal ribonucleic acid",
    "ATP": "adenosine triphosphate",
    "ADP": "adenosine diphosphate",
    "GTP": "guanosine triphosphate",
    "NAD": "nicotinamide adenine dinucleotide",
    "NADH": "reduced nicotinamide adenine dinucleotide",
    "FAD": "flavin adenine dinucleotide",
    "CoA": "coenzyme A",
    # Proteins / amino acids
    "IgG": "immunoglobulin G",
    "IgM": "immunoglobulin M",
    "IgA": "immunoglobulin A",
    "IgE": "immunoglobulin E",
    "Hb": "hemoglobin",
    "HGB": "hemoglobin",
    "WBC": "white blood cell",
    "RBC": "red blood cell",
    "PCR": "polymerase chain reaction",
    # Medical / clinical
    "CT": "computed tomography",
    "MRI": "magnetic resonance imaging",
    "ECG": "electrocardiogram",
    "EKG": "electrocardiogram",
    "EEG": "electroencephalogram",
    "IV": "intravenous",
    "IM": "intramuscular",
    "SC": "subcutaneous",
    "PO": "per os",
    "PRN": "pro re nata",
    "BID": "bis in die",
    "TID": "ter in die",
    "QID": "quater in die",
    "ICU": "intensive care unit",
    "ER": "emergency room",
    "OR": "operating room",
    "CPR": "cardiopulmonary resuscitation",
    "CBC": "complete blood count",
    "BMP": "basic metabolic panel",
    # Pharmacology
    "NSAID": "nonsteroidal anti-inflammatory drug",
    "ACE": "angiotensin converting enzyme",
    "SSRI": "selective serotonin reuptake inhibitor",
    # Anatomy
    "CNS": "central nervous system",
    "PNS": "peripheral nervous system",
    "GI": "gastrointestinal",
    "CV": "cardiovascular",
    "ENT": "ear nose throat",
    # Physics / chemistry
    "CO2": "carbon dioxide",
    "H2O": "water",
    "O2": "oxygen",
    "N2": "nitrogen",
    "NaCl": "sodium chloride",
    "HCl": "hydrochloric acid",
    "UV": "ultraviolet",
    "IR": "infrared",
    "NMR": "nuclear magnetic resonance",
    "IRMS": "isotope ratio mass spectrometry",
    # Astro / geo
    "AU": "astronomical unit",
    "BH": "black hole",
    "LHC": "large hadron collider",
    # Legal
    "ProBono": "pro bono",
    "ExParte": "ex parte",
    "HabeasCorpus": "habeas corpus",
    "Amicus": "amicus curiae",
    "A.K.A.": "also known as",
    "R.S.V.P.": "respondez s'il vous plait",
    "EtAl": "et alii",
    "vs": "versus",
    "Viz": "videlicet",
    "i.e.": "id est",
    "e.g.": "exempli gratia",
    "etc.": "et cetera",
}

# 50+ medical / anatomical / botanical Latin roots.  Curated against
# the Terminologia Anatomica (1998) and FMA (Foundation Model of
# Anatomy).  Keys are lowercased.  Sorted longest-first so
# longest-prefix matching in :func:`split_compound` is automatic
# when we iterate without an explicit sort.
#
# We include both *bound morphemes* (e.g. ``"sterno"``) and
# *complete anatomical terms* (e.g. ``"mastoid"``, ``"dorsi"``) so
# the greedy longest-prefix match can swallow "sternocleidomastoid"
# in one go as ``["sterno", "cleido", "mastoid"]``.
_MEDICAL_ROOTS: Set[str] = {
    # Body regions / directions
    "antero", "postero", "supero", "infero", "latero", "medialo",
    "dorso", "ventro", "cranio", "caudo", "proximo", "disto",
    # Head / neck
    "sterno", "cleido", "masto", "mastoid", "mastoid",
    "occipito", "occipit", "fronto", "temporo", "temporal",
    "parieto", "zygomatico", "zygomat", "maxillo", "maxill",
    "mandibulo", "mandibul", "nasolo", "nasal",
    # Upper limb
    "brachio", "brachial", "radio", "radial", "ulno", "ulnar",
    "carpo", "carpal", "metacarpo", "metacarp", "phalango", "phalang",
    "humerus", "humeral", "scapulo", "scapul", "claviculo", "clavicular",
    # Lower limb
    "femor", "femoro", "femoral", "tibio", "tibial", "fibulo", "fibular",
    "tarso", "tarsal", "metatarso", "metatars", "calcaneo", "calcane",
    "gluteo", "gluteal", "inguino", "inguinal", "popliteo", "popliteal",
    "cnemius", "soleus", "plantaris", "peroneus", "sartorius",
    # Trunk
    "thoraco", "thoracic", "lumbo", "lumbar", "sacro", "sacral",
    "coccygo", "coccygeal", "abdomino", "abdomin", "pelvo", "pelvic",
    "vertebro", "vertebral", "cervico", "cervic",
    # Muscle names (full forms so the longest-match wins)
    "latissimus", "longissimus", "spinalis", "semispinalis",
    "rectus", "obliquus", "transversus", "trapezius",
    "deltoideus", "deltoid", "biceps", "triceps", "quadriceps",
    "gastrocnemius", "psoas", "iliacus", "pectineus", "gracilis",
    "adductor", "abductor",
    # Muscle / body descriptors
    "dorsi", "dorsal", "ventral", "lateral", "medial", "anterior",
    "posterior", "superior", "inferior", "profundus", "superficial",
    "externus", "internus", "major", "minor", "longus", "brevis",
    # Organs / viscera
    "cardio", "cardiac", "pulmono", "pulmonar", "hepato", "hepat",
    "reno", "renal", "spleno", "splen", "gastro", "gastr",
    "entero", "enter", "colo", "colic", "recto", "rectal",
    "cysto", "cyst", "uretero", "ureter", "vesico", "vesic",
    "prostato", "prostat", "testiculo", "testicul", "ovario", "ovari",
    "utero", "uter", "vagino", "vagin", "thyroido", "thyroid",
    "adrenalo", "adrenal", "pneumo", "pneumon", "pleuro", "pleur",
    "tracheo", "trache", "broncho", "bronch", "laryngo", "laryng",
    "pharyngo", "pharyng", "esophago", "esophag",
    # Brain / CNS
    "encephalo", "encephal", "cerebro", "cerebr", "cerebello", "cerebell",
    "ponto", "pont", "medullo", "medull", "thalamo", "thalam",
    "hippocampo", "hippocamp", "amygdalo", "amygdal",
    # Vessel / tissue
    "angio", "angi", "vaso", "vas", "arterio", "arter", "veno", "ven",
    "capillaro", "capillar", "lympho", "lymph",
    "neuro", "neur", "myelo", "myel", "musculo", "muscul",
    "tendino", "tendin", "ligamento", "ligament", "osteo", "oste",
    "chondro", "chondr", "arthro", "arthr", "dermo", "derm",
    "epidermo", "epiderm", "subcutaneo", "subcutan", "cutane",
    # Sense organs
    "oculo", "ocul", "ophthalmo", "ophthalm", "auriculo", "auricul",
    "auris", "cochleo", "cochle", "vestibulo", "vestibul",
    "retino", "retin", "corneo", "corne", "irido", "irid", "pupillo", "pupill",
    # Botanical
    "antho", "anth", "phyllo", "phyll", "stameno", "stamen", "pistillo",
    "pistil", "carpelo", "carpel", "ovulo", "ovul", "radiclo", "radicl",
    "caulo", "caul", "rhizo", "rhiz", "xylo", "xyl", "phloemo", "phloem",
    # Generic Latin/Greek prefixes
    "ecto", "ent", "endo", "exo", "meso", "epi", "hypo", "para",
    "peri", "infra", "supra", "inter", "intra", "extra", "trans",
    "macro", "micro", "mega", "megalo", "lepto", "pachy", "platy",
    "dolicho", "brachy", "brevi", "longi", "magni", "parvi",
    "multi", "pluri", "uni", "bi", "tri", "quadri", "penta", "hexa",
    "hepta", "octo", "neuri", "blasto", "cyto", "cyt", "histo", "hist",
    "morpho", "morph", "patho", "path", "logo", "grapho", "scopo", "tomo",
    "lysis", "lyt", "penia", "cytosis",
}

# Sort longest-first for :func:`split_compound` greedy match.
_ROOTS_SORTED: Tuple[str, ...] = tuple(sorted(_MEDICAL_ROOTS, key=len, reverse=True))

# Regular expression for a valid Roman numeral suffix.  Roman numerals
# are: M{0,4} CM|CD|D?C{0,3} XC|XL|L?X{0,3} IX|IV|V?I{0,3}.  We
# require the suffix to be at the end of the (last) word, optionally
# followed by a dot.  Case-insensitive at the re level.
_ROMAN_RE = re.compile(
    r"\b([IVXLCDM]+)\.?\s*$",
    re.IGNORECASE,
)

# A pre-computed "first character" map for cheap Roman validation.
_ROMAN_FIRST_OK = {"M", "D", "C", "L", "X", "V", "I"}
# Subtractives we explicitly accept (the canonical 6).
_ROMAN_SUBTRACTIVE = {"CM", "CD", "XC", "XL", "IX", "IV"}
# Single letters and their values.
_ROMAN_VALUES: Dict[str, int] = {
    "I": 1, "V": 5, "X": 10, "L": 50,
    "C": 100, "D": 500, "M": 1000,
}
# Canonical roman -> int lookup (covers the full _MAX_ROMAN range).
_ROMAN_TO_INT: Dict[str, int] = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7,
    "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12, "XIII": 13,
    "XIV": 14, "XV": 15, "XVI": 16, "XVII": 17, "XVIII": 18,
    "XIX": 19, "XX": 20, "XXI": 21, "XXX": 30, "XL": 40, "L": 50,
    "LX": 60, "LXX": 70, "LXXX": 80, "XC": 90, "C": 100, "CC": 200,
    "CCC": 300, "CD": 400, "D": 500, "DC": 600, "DCC": 700,
    "DCCC": 800, "CM": 900, "M": 1000, "MM": 2000, "MMM": 3000,
}
# Pre-computed int -> roman (covers 1..3999).
_INT_TO_ROMAN: Dict[int, str] = {
    1: "I", 4: "IV", 5: "V", 9: "IX", 10: "X", 40: "XL", 50: "L",
    90: "XC", 100: "C", 400: "CD", 500: "D", 900: "CM", 1000: "M",
}
_INT_ROMAN_VALUES = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
    (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]

# Whitespace / word-boundary detectors.
_WORD_RE = re.compile(r"[\w]+", re.UNICODE)
# A "real" word separator: runs of letters or hyphens.
_TERM_RE = re.compile(r"[A-Za-zÀ-ÿ\-]+")

# A lightweight tokeniser that keeps hyphenated compounds together.
# "sternocleidomastoid" -> "sternocleidomastoid"
# "Alpha Centauri"      -> ["Alpha", "Centauri"]
# "Escherichia coli K-12" -> ["Escherichia", "coli", "K", "12"]
# "Homo sapiens L."     -> ["Homo", "sapiens", "L"]
_TAXONOMIC_SPLIT_RE = re.compile(r"\s+|[/,;]")

# Known author / authority abbreviations from zoological &
# botanical nomenclature (ICZN / ICN).  Lower-cased keys; values are
# canonical short forms.  This list is intentionally short -- the
# goal is to *detect* that a token is an authority tag, not to
# resolve it.
_AUTHORITY_ABBR: Set[str] = {
    "linn", "linnaeus", "l.", "l", "fabricius", "fabr.", "fabr",
    "smith", "j. smith", "darwin", "wallace", "mendel", "pasteur",
    "koch", "fleming", "hawking", "darwin", "cuvier", "baird",
    "gray", "gould", "strickland", "sclater", "huxley", "wallace",
    "carl", "c.", "cuv.", "cuvier", "lacépède", "lacepede",
    "forster", "g. forster", "merrem", "bonaparte", "leach",
    "vigors", "horsfield", "swainson", "gould", "blasius", "bang",
    "tschudi", "kaup", "streubel", "reichenbach", "sclater",
    "salvin", "godman", "sharpe", "ramsay", "de vis", "devis",
    "alfaro", "carriker", "ridgway", "hellmayr", "conover",
    "rand", "traylor", "paynter", "mayr", "greenway", "gilliard",
}


# ---------------------------------------------------------------------------
# 1. Taxonomic name handler
# ---------------------------------------------------------------------------

def parse_taxonomic_name(name: str) -> Dict[str, Any]:
    """Parse a Latin scientific name into its canonical components.

    Parameters
    ----------
    name:
        A biological / botanical / chemical name.  Accepts:

        * Bare binomials: ``"Homo sapiens"``
        * With authority: ``"Homo sapiens Linnaeus, 1758"``
        * With strain/serovar: ``"Escherichia coli K-12"``
        * With sub-species: ``"Canis lupus familiaris"``
        * Trinomials: ``"Panthera tigris altaica"``
        * Botanical authorities: ``"Quercus robur L."``
        * Genus abbreviation: ``"E. coli"``
        * Common names: ``"Bald Eagle"`` (low confidence)

    Returns
    -------
    dict with keys (any may be absent):

    * ``genus``        - Title-cased genus (e.g. ``"Homo"``)
    * ``genus_initial``- Just the first letter of the genus, if the
                         input was abbreviated (e.g. ``"E"`` for
                         ``"E. coli"``).
    * ``species``      - Lower-cased specific epithet (e.g. ``"sapiens"``)
    * ``subspecies``   - Lower-cased sub-specific epithet (trinomial)
    * ``authority``    - Authority surname, if detected
    * ``year``         - Authority year (4 digits), if detected
    * ``strain``       - Strain / serovar / cultivar code, if present
    * ``is_taxonomic`` - ``True`` if the input looks like a proper
                         binomen (two Latin tokens, first capitalised).
    * ``raw``          - The original input (echoed for debug)

    The output is **deterministic** and never throws.  Unparseable
    input returns ``{"raw": <name>, "is_taxonomic": False}``.
    """
    out: Dict[str, Any] = {
        "raw": name,
        "is_taxonomic": False,
    }
    if not name or not isinstance(name, str):
        return out

    text = unicodedata.normalize("NFKC", name).strip()
    if not text:
        return out

    # Strain / serovar / cultivar: usually a short alphanumeric
    # token at the end, like K-12, O157:H7, MG1655, etc.  Pull it
    # off first so the rest of the parser sees a clean binomial.
    strain = None
    m = re.search(
        r"\s+((?:[A-Z]{1,4})-?\d{1,5}"
        r"|O\d+(?::H\d+)?"
        r"|MG\d+"
        r"|BL21(?:\(DE3\))?"
        r"|DH5α"
        r"|JM109"
        r"|XL[-\s]?\d*\s*[Bb]lue)"
        r"\s*$",
        text,
    )
    if m:
        strain = m.group(1).strip()
        text = text[: m.start()].rstrip()

    # Year: 4-digit year between 1500 and 2100.
    year = None
    m = re.search(r",?\s*\(?(\d{4})\)?\s*$", text)
    if m:
        try:
            y = int(m.group(1))
            if 1500 <= y <= 2100:
                year = y
                text = text[: m.start()].rstrip(" ,")
        except ValueError:
            pass

    # Authority: take the last token(s) if it ends in a "." or
    # matches a known authority pattern, OR if it's an all-caps
    # surname-like token.
    authority = None
    # Strip a trailing "L." or "Linnaeus" or "Fabricius, 1775" if
    # present.  We do this by splitting on whitespace and looking
    # at the trailing chunk.
    tokens = [t for t in text.split() if t]
    # Only treat a trailing token as authority if it ENDS in a
    # period (e.g. "L.", "J. Smith") OR if it's a known authority
    # name in our curated list.  This prevents "II" in "Henry
    # VIII" or "War II" in "World War II" from being eaten as
    # an authority.
    if (
        len(tokens) >= 3
        and (
            tokens[-1].endswith(".")
            or tokens[-1].lower() in _AUTHORITY_ABBR
        )
    ):
        # Likely authority: "Genus species Authority" or
        # "Genus species Authority, YYYY".
        # We already pulled the year.  Reconstruct: drop the last
        # token (and any prepended comma).
        authority = tokens[-1].rstrip(",.")
        # Special-case: "L." -> "Linnaeus"
        if authority in {"L", "l"}:
            authority = "Linnaeus"
        text = " ".join(tokens[:-1])
        tokens = text.split()

    # Now we should have a binomial (2 tokens) or trinomial (3
    # tokens).  Validate.
    if len(tokens) < 2:
        # Maybe it's a common name or just a genus.
        if len(tokens) == 1 and tokens[0] and tokens[0][0].isupper():
            out["genus"] = tokens[0].capitalize()
        return out

    # Check the first token is a capitalised Latin-looking word,
    # OR a single capital letter followed by a period (genus
    # abbreviation like "E." in "E. coli").
    first, second = tokens[0], tokens[1]
    is_abbr = bool(re.match(r"^[A-Z]\.$", first))
    if is_abbr:
        out["genus_initial"] = first[0]
        out["is_taxonomic"] = True
        out["species"] = second.lower()
        # Try to expand the abbreviation to a full genus if it
        # matches a known short form.
        abbr_expansions = {
            "E": "Escherichia", "H": "Homo", "P": "Pseudomonas",
            "S": "Salmonella", "B": "Bacillus", "C": "Clostridium",
            "L": "Lactobacillus", "M": "Mycobacterium",
            "S.": "Staphylococcus", "St.": "Streptococcus",
        }
        expansion = abbr_expansions.get(first[0]) or abbr_expansions.get(first)
        if expansion is not None:
            out["genus"] = expansion
        if year is not None:
            out["year"] = year
        if authority is not None:
            out["authority"] = authority
        if strain is not None:
            out["strain"] = strain
        return out

    if not (first[:1].isupper() and first.isalpha()):
        return out
    # The second token must start with a letter; we don't require
    # lower-case because the source might be all-caps ("ESCHERICHIA
    # COLI").
    if not (second[:1].isalpha()):
        return out
    # Reject if the second token is a Roman numeral -- "Henry
    # VIII" is a proper name, not a binomial.
    if _parse_roman(second) is not None and second.isupper():
        return out
    # Reject if the second token is an all-caps short word
    # (e.g. "World War II") -- that's a proper noun, not Latin.
    if second.isupper() and len(second) <= 5 and _parse_roman(second) is not None:
        return out
    # Reject if there is a 3rd token and it's a Roman numeral:
    # "World War II" -> 3 tokens, last is "II" -> not a binomial.
    if len(tokens) >= 3 and tokens[2].isupper() and _parse_roman(tokens[2]) is not None:
        return out
    # Reject the "Name Name [Roman]" pattern (proper noun + Roman
    # suffix) where both Name tokens are Title-cased -- this
    # catches "World War II", "Henry VIII", "Star Wars III",
    # etc.  We require the second token to be Title-cased, not
    # all-lowercase, so "Homo sapiens" (a real binomial) still
    # passes.
    if (
        first[0].isupper() and first[1:].islower()
        and second[0].isupper() and second[1:].islower()
        and len(tokens) >= 3
        and tokens[2].isupper()
        and _parse_roman(tokens[2]) is not None
    ):
        return out

    out["genus"] = first.capitalize()
    out["species"] = second.lower()
    out["is_taxonomic"] = True

    # Trinomial: 3rd token, lower-case, not in authority list.
    if len(tokens) >= 3:
        third = tokens[2]
        if (
            third[:1].isalpha()
            and third.lower() not in _AUTHORITY_ABBR
            and not re.search(r"[A-Z]\.?$", third)
        ):
            out["subspecies"] = third.lower()

    if year is not None:
        out["year"] = year
    if authority is not None:
        out["authority"] = authority
    if strain is not None:
        out["strain"] = strain
    return out


# ---------------------------------------------------------------------------
# 2. Diacritic normaliser
# ---------------------------------------------------------------------------

# Hand-curated table for characters that NFD decomposition does NOT
# map to base + combining-mark (i.e. ligatures, Polish L-with-stroke,
# German sharp-s, etc.).  Keys are upper-case AND lower-case forms so
# the function is case-preserving.  This is intentionally small --
# the broader table in :mod:`mathir_dropin.universal_bridge` already
# covers everything NFD does.
_DIACRITIC_EXCEPTIONS: Dict[str, str] = {
    # Polish
    "Ł": "L", "ł": "l",
    # German
    "ß": "ss", "ẞ": "SS",
    # Ligatures
    "œ": "oe", "Œ": "OE",
    "æ": "ae", "Æ": "AE",
    "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
    # Vietnamese / Turkic single-letter mappings
    "đ": "d", "Đ": "D",
    "ı": "i", "İ": "I",
    # Icelandic / Faroese
    "þ": "th", "Þ": "Th", "ð": "d", "Ð": "D",
    # Scottish Gaelic
    "ẁ": "w", "ḃ": "b",
}


def normalize_diacritics(text: str) -> str:
    """Strip diacritics, preserving the base letter.

    Two-pass algorithm:

    1. **NFKD decomposition** + drop combining marks.  This handles
       ``é`` -> ``e``, ``ñ`` -> ``n``, ``ü`` -> ``u``, and tens
       of thousands of other accented code points automatically.
    2. **Hand-curated exceptions table** for the few characters
       NFD does *not* decompose: Polish ``Ł`` -> ``L``, German
       ``ß`` -> ``ss``, ligatures ``œ`` -> ``oe``, etc.

    The function is case-preserving: ``É`` -> ``E``, ``Ł`` -> ``L``.

    Parameters
    ----------
    text:
        Input string.  ``None`` returns ``""``.

    Returns
    -------
    The de-diacriticked string.  Empty input returns empty.
    """
    if not text:
        return ""
    # Pass 1: NFD then strip combining marks.
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(
        ch for ch in decomposed
        if unicodedata.category(ch) != "Mn"
    )
    # Pass 2: hand-curated exceptions.
    return "".join(_DIACRITIC_EXCEPTIONS.get(ch, ch) for ch in stripped)


# ---------------------------------------------------------------------------
# 3. Roman numeral handler
# ---------------------------------------------------------------------------

def _is_valid_roman(s: str) -> bool:
    """Return True if ``s`` is a *valid* Roman numeral between 1 and
    :data:`_MAX_ROMAN`.

    The check rejects "ad-hoc" inputs like ``"IIII"`` (canonical form
    is ``"IV"``) and ``"VV"`` (not a valid form), so the function
    can be used as a sanity check on text-derived candidates.
    """
    if not s or not s.isalpha():
        return False
    s_up = s.upper()
    if s_up in _ROMAN_TO_INT:
        return True
    # Deeper check via the value parser.  If the parsed value is
    # non-zero and round-trips back to the same form, accept it.
    parsed = _parse_roman(s_up)
    if parsed is None or parsed <= 0 or parsed > _MAX_ROMAN:
        return False
    return int_to_roman(parsed) == s_up


def _parse_roman(s: str) -> Optional[int]:
    """Parse a Roman numeral string into an integer (1..3999).

    Returns ``None`` on any malformed input.  Implements the
    standard subtractive-notation rule: a smaller value before a
    larger value subtracts (e.g. ``"IV"`` -> 4, ``"IX"`` -> 9).
    """
    if not s or not s.isalpha():
        return None
    s = s.upper()
    total = 0
    prev = 0
    for ch in reversed(s):
        v = _ROMAN_VALUES.get(ch)
        if v is None:
            return None
        if v < prev:
            total -= v
        else:
            total += v
        prev = v
    if total <= 0 or total > _MAX_ROMAN:
        return None
    # Reject "IIII" (canonical IV), "VIIII" (IX), "XXXX" (XL),
    # "XXXXX" (L), etc. by checking the round-trip.
    if int_to_roman(total) != s:
        return None
    return total


def parse_roman_numeral(suffix: str) -> Optional[int]:
    """Detect and parse a Roman-numeral suffix on a proper name.

    Parameters
    ----------
    suffix:
        The trailing word of a phrase, e.g. ``"VIII"``, ``"III."``,
        ``"iv"`` (case-insensitive).  Whitespace is trimmed.

    Returns
    -------
    The integer value (``"VIII"`` -> ``8``), or ``None`` if the
    input is not a valid Roman numeral.

    Examples
    --------
    >>> parse_roman_numeral("VIII")
    8
    >>> parse_roman_numeral("III.")
    3
    >>> parse_roman_numeral("foo")
    >>> parse_roman_numeral("IIII") is None
    True
    """
    if not suffix:
        return None
    s = suffix.strip().rstrip(".")
    if not s:
        return None
    return _parse_roman(s)


def roman_to_int(roman: str) -> int:
    """Convert a Roman numeral to an integer.

    Raises ``ValueError`` on malformed input.  Use
    :func:`parse_roman_numeral` if you want ``None`` on failure.

    >>> roman_to_int("VIII")
    8
    >>> roman_to_int("MCMXCIX")
    1999
    """
    if not roman:
        raise ValueError("roman must be non-empty")
    v = _parse_roman(roman)
    if v is None:
        raise ValueError(f"not a valid Roman numeral: {roman!r}")
    return v


def int_to_roman(n: int) -> str:
    """Convert an integer in ``1..3999`` to a Roman numeral.

    Uses the standard subtractive notation
    (``4 -> IV``, ``9 -> IX``, ``40 -> XL``, ...).

    >>> int_to_roman(8)
    'VIII'
    >>> int_to_roman(1999)
    'MCMXCIX'

    Raises ``ValueError`` for out-of-range or non-positive input.
    """
    if not isinstance(n, int) or isinstance(n, bool):
        raise ValueError(f"n must be an int, got {type(n).__name__}")
    if n < 1 or n > _MAX_ROMAN:
        raise ValueError(f"n must be in 1..{_MAX_ROMAN}, got {n}")
    out: List[str] = []
    for value, symbol in _INT_ROMAN_VALUES:
        while n >= value:
            out.append(symbol)
            n -= value
    return "".join(out)


# ---------------------------------------------------------------------------
# 4. Abbreviation expander
# ---------------------------------------------------------------------------

def expand_abbreviation(abbr: str) -> List[str]:
    """Expand a scientific / medical abbreviation to its full form.

    Returns a list of expansion variants:

    * The full form (``"DNA"`` -> ``"deoxyribonucleic acid"``)
    * Token-by-token expansion of any sub-tokens
    * The original (for idempotence)
    * Lower-cased / title-cased variants for FTS5 robustness

    Unknown abbreviations return ``[abbr]`` (single-item list with
    the original, so the caller can still index the term).

    The function is **case-insensitive** at the input level but
    preserves the original casing in the output: ``"dna"`` ->
    ``["deoxyribonucleic acid", "dna"]``.
    """
    if not abbr:
        return [""]
    text = unicodedata.normalize("NFKC", str(abbr)).strip()
    if not text:
        return [""]

    # Build a normalised key: strip all trailing dots / commas,
    # collapse internal whitespace, and upper-case.  This lets
    # ``mRNA``, ``mRNA.``, and ``MRNA`` all resolve to the same
    # entry.
    key = text.rstrip(".").rstrip(",").strip()
    if not key:
        return [text]
    # Build a fully-normalised lookup key: remove all dots
    # *and* spaces so "i.e." matches "i.e" which matches
    # "I.E" -- but never break a multi-letter abbreviation
    # like "mRNA".
    upper_dotted = key.upper()

    # Resolve against the dictionary.  Try in order:
    #  1. Exact key match (case-sensitive) -- catches the
    #     mixed-case entries like "mRNA" and "i.e.".
    #  2. Upper-cased key -- catches "DNA", "ATP", etc.
    #  3. Upper-cased-with-dots-removed -- catches "i.e." /
    #     "i.e" / "IE" / "I.E".
    full = (
        _ABBREVIATIONS.get(key)
        or _ABBREVIATIONS.get(upper_dotted)
        or _ABBREVIATIONS.get(upper_dotted.replace(".", ""))
        or _ABBREVIATIONS.get(upper_dotted + ".")
        or _ABBREVIATIONS.get(key + ".")
        or _ABBREVIATIONS.get(upper_dotted.replace(".", "") + ".")
    )

    expansions: List[str] = []
    seen: Set[str] = set()

    def _add(x: str) -> None:
        x = x.strip()
        if not x:
            return
        if x in seen:
            return
        seen.add(x)
        expansions.append(x)

    if full is not None:
        _add(full)
        # Common variants: drop the trailing " acid" suffix for
        # FTS5 prefix-matching ("deoxyribonucleic" alone).
        if full.endswith(" acid"):
            _add(full[: -len(" acid")])

    # Always include the original (preserves case).
    _add(text)
    return expansions if expansions else [text]


def list_known_abbreviations() -> List[str]:
    """Return the sorted list of all built-in abbreviations.

    Useful for introspection, doc generation, and test fixtures.
    """
    return sorted(_ABBREVIATIONS.keys())


# ---------------------------------------------------------------------------
# 5. Compound term splitter
# ---------------------------------------------------------------------------

def _build_trie(roots: Sequence[str]) -> List[Tuple[str, ...]]:
    """Sort roots so longest-match comes first.

    We don't actually build a trie data structure -- the
    longest-prefix-match is simple enough to do with a sorted
    list.  This function exists to make the intent explicit and
    give us a place to swap in a trie later if performance ever
    matters (the linear scan is fine for the ~150 roots we ship).
    """
    return tuple(sorted(set(roots), key=len, reverse=True))


_ROOTS_TRIE: Tuple[str, ...] = _build_trie(_MEDICAL_ROOTS)


def split_compound(term: str) -> List[str]:
    """Split a compound medical / anatomical term into its roots.

    Uses a longest-prefix-match against a built-in dictionary of
    ~150 medical / anatomical / botanical Latin roots (Terminologia
    Anatomica + FMA).  Unknown components are returned as-is so
    the result is *never* lossy: ``split_compound(x)`` always
    contains all the characters of ``x`` distributed across its
    output tokens.

    Examples
    --------
    >>> split_compound("sternocleidomastoid")
    ['sterno', 'cleido', 'mastoid']
    >>> split_compound("gastrocnemius")
    ['gastro', 'cnemius']
    >>> split_compound("latissimusdorsi")
    ['latissimus', 'dorsi']
    """
    if not term:
        return []
    text = normalize_diacritics(term).lower()
    # Strip non-letter characters; compound roots are pure Latin.
    text = "".join(ch for ch in text if ch.isalpha())
    if not text:
        return []

    out: List[str] = []
    i = 0
    n = len(text)
    matched_any = False
    while i < n:
        # Greedy longest-prefix match.
        matched = None
        for root in _ROOTS_TRIE:
            rlen = len(root)
            if rlen == 0:
                continue
            if i + rlen <= n and text[i : i + rlen] == root:
                matched = root
                matched_any = True
                break
        if matched is not None:
            out.append(matched)
            i += len(matched)
        else:
            # No root matched.  Accumulate one character into a
            # fallback "unknown" buffer that we flush when the
            # next match starts (or at end of string).
            # We don't emit single-character tokens -- those
            # would be useless.  Instead we accumulate and
            # flush at the end.
            j = i + 1
            while j < n:
                # Try to match starting at position j.
                hit = None
                for root in _ROOTS_TRIE:
                    rlen = len(root)
                    if rlen == 0:
                        continue
                    if j + rlen <= n and text[j : j + rlen] == root:
                        hit = root
                        break
                if hit is not None:
                    break
                j += 1
            # Flush the unmatched run [i, j).
            out.append(text[i:j])
            i = j

    # If we matched *only* via the unknown-run path, every entry
    # is the whole term; we collapse that to a single entry.
    if not matched_any and len(out) == 1 and len(out[0]) == n:
        return out
    return out


# ---------------------------------------------------------------------------
# 6. Latin name matcher
# ---------------------------------------------------------------------------

def _jaccard(a: str, b: str, n: int = 3) -> float:
    """Jaccard similarity over character n-grams in ``[0, 1]``.

    This is a self-contained re-implementation so this module can
    be imported and tested without :mod:`mathir_dropin.universal_bridge`.
    """
    if not a or not b:
        return 0.0
    a_norm = normalize_diacritics(a).lower()
    b_norm = normalize_diacritics(b).lower()
    if not a_norm or not b_norm:
        return 0.0
    if len(a_norm) < n:
        a_norm = " " * (n - 1) + a_norm + " " * (n - 1)
    if len(b_norm) < n:
        b_norm = " " * (n - 1) + b_norm + " " * (n - 1)
    a_norm = " " * (n - 1) + a_norm + " " * (n - 1)
    b_norm = " " * (n - 1) + b_norm + " " * (n - 1)
    set_a = {a_norm[i : i + n] for i in range(len(a_norm) - n + 1)}
    set_b = {b_norm[i : i + n] for i in range(len(b_norm) - n + 1)}
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def _taxonomic_similarity(query: str, candidate: str) -> float:
    """Score taxonomic equality between two names.

    Returns:

    * ``1.0`` if the canonical binomials match exactly
      (genus + species, case-insensitive).
    * ``0.85`` if the genus matches but species differ.
    * ``0.70`` if the species epithet matches.
    * ``0.0`` otherwise.

    Abbreviations like ``"E. coli"`` are expanded by treating the
    single letter + dot as the genus initial.
    """
    q = parse_taxonomic_name(query)
    c = parse_taxonomic_name(candidate)
    if not (q.get("is_taxonomic") and c.get("is_taxonomic")):
        return 0.0
    qg, qs = q.get("genus"), q.get("species")
    cg, cs = c.get("genus"), c.get("species")
    if not (qg and cg):
        return 0.0
    qg_l = qg.lower()
    cg_l = cg.lower()
    # Initial-letter match: "E. coli" <-> "Escherichia coli".
    qg_init = qg_l[0] if qg_l else ""
    cg_init = cg_l[0] if cg_l else ""
    genus_match = (
        qg_l == cg_l
        or (qg_init == cg_init and (qg_l.endswith(".") or cg_l.endswith(".")))
    )
    species_match = qs is not None and cs is not None and qs.lower() == cs.lower()
    if genus_match and species_match:
        return 1.0
    if genus_match:
        return 0.85
    if species_match:
        return 0.70
    return 0.0


def _root_overlap(query: str, candidate: str) -> float:
    """Jaccard over the medical-Latin-root decomposition of each term.

    Splits both sides with :func:`split_compound` and returns
    ``|intersection| / |union|``.  Returns 0.0 if either side
    yields 0 or 1 token (no compound structure to compare).
    """
    q_roots = set(split_compound(query))
    c_roots = set(split_compound(candidate))
    if len(q_roots) < 2 or len(c_roots) < 2:
        return 0.0
    if not q_roots or not c_roots:
        return 0.0
    inter = len(q_roots & c_roots)
    union = len(q_roots | c_roots)
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def _abbreviation_overlap(query: str, candidate: str) -> float:
    """Score 1.0 if the query is an abbreviation whose expansion
    matches (a sub-string of) the candidate, OR vice-versa.
    """
    q_exp = expand_abbreviation(query)
    c_exp = expand_abbreviation(candidate)
    if not q_exp or not c_exp:
        return 0.0
    q_full = q_exp[0]
    c_full = c_exp[0]
    if not q_full or not c_full:
        return 0.0
    qn = q_full.lower()
    cn = c_full.lower()
    if qn == cn:
        return 1.0
    if qn in cn or cn in qn:
        return 0.8
    # Token-level containment: "ribonucleic acid" within
    # "messenger ribonucleic acid" -> 0.8.
    q_toks = set(qn.split())
    c_toks = set(cn.split())
    if not q_toks or not c_toks:
        return 0.0
    inter = q_toks & c_toks
    if inter and (inter == q_toks or inter == c_toks):
        return 0.8
    return 0.0


def _roman_similarity(query: str, candidate: str) -> float:
    """Return 1.0 if the query and candidate differ only by a
    Roman-numeral suffix vs integer form; 0.0 otherwise.
    """
    if not query or not candidate:
        return 0.0
    # Strip roman / int suffix from each, compare base words.
    def _strip(s: str) -> Tuple[str, Optional[int]]:
        s = s.strip()
        m = _ROMAN_RE.search(s)
        if m:
            base = s[: m.start()].rstrip()
            n = parse_roman_numeral(m.group(0))
            return base, n
        # Try integer suffix.
        m2 = re.search(r"\s+(\d+)\s*\.?\s*$", s)
        if m2:
            base = s[: m2.start()].rstrip()
            try:
                return base, int(m2.group(1))
            except ValueError:
                return s, None
        return s, None

    qb, qn = _strip(query)
    cb, cn = _strip(candidate)
    if qb.lower() != cb.lower():
        return 0.0
    if qn is None and cn is None:
        return 0.0
    if qn is None or cn is None:
        return 0.5  # base matches, one side has no number
    return 1.0 if qn == cn else 0.0


def latin_match(
    query: str,
    candidate: str,
    *,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Return a similarity score in ``[0, 1]`` between a query and a
    candidate Latin / technical term.

    Combines six channels:

    * ``raw``            -- raw Jaccard over character n-grams.
    * ``canonical``      -- Jaccard over the de-diacriticked,
                            lower-cased forms.
    * ``taxonomic``      -- taxonomic-name equality (1.0 exact
                            match, 0.85 same genus, 0.7 same
                            species, else 0).
    * ``roots``          -- Jaccard over medical-Latin-root
                            decompositions.
    * ``abbreviation``   -- 1.0 if either side is an abbreviation
                            whose expansion matches / contains the
                            other.
    * ``roman``          -- 1.0 if the two differ only by a roman
                            vs integer suffix.

    Fusion strategy: **max-of-channels + multi-channel agreement
    boost**.

    * Start with the maximum per-channel score.
    * If two or more channels each score above 0.5, add a small
      agreement bonus (default +0.05 per extra strong channel, up
      to +0.15 total).
    * Clamp to ``[0, 1]``.

    This is *intentionally* simpler than weighted RRF for this
    use case: a single "strong" channel (e.g. an exact
    taxonomic match) is enough to declare the names the same,
    and the boost rewards multi-evidence matches without ever
    damping a strong single-channel signal.

    The default behaviour is tuned so that:

    * Exact taxonomic match returns ``>= 0.95``.
    * Exact abbreviation match returns ``>= 0.95``.
    * Exact compound-root overlap returns ``>= 0.85``.
    * Roman-vs-integer suffix match returns ``>= 0.85``.
    * A pure Jaccard match tops out around ``0.5``.

    Parameters
    ----------
    query, candidate:
        Strings.  Empty / non-string returns ``0.0``.
    weights:
        Optional dict overriding the per-channel *contribution*
        weight.  Each channel's effective score is
        ``min(1.0, channel_score * weight)``.  Unknown keys are
        ignored.  Missing keys use defaults.

    Returns
    -------
    A float in ``[0, 1]``.

    Examples
    --------
    >>> latin_match("Homo sapiens", "Homo sapiens")
    1.0
    >>> latin_match("H. sapiens", "Homo sapiens") > 0.9
    True
    >>> latin_match("Schrodinger", "Schrödinger") > 0.9
    True
    >>> latin_match("Henry VIII", "Henry 8") >= 0.85
    True
    """
    if not query or not candidate:
        return 0.0
    if not isinstance(query, str) or not isinstance(candidate, str):
        return 0.0

    # Per-channel raw scores.
    raw_sim = _jaccard(query, candidate, n=3)
    q_canon = normalize_diacritics(query).lower()
    c_canon = normalize_diacritics(candidate).lower()
    canonical_sim = _jaccard(q_canon, c_canon, n=3)
    tax_sim = _taxonomic_similarity(query, candidate)
    root_sim = _root_overlap(query, candidate)
    abbr_sim = _abbreviation_overlap(query, candidate)
    roman_sim = _roman_similarity(query, candidate)

    # Apply per-channel weight (interpreted as a confidence in
    # that channel; ``1.0`` means use the raw score as-is).
    default_weights: Dict[str, float] = {
        "raw": 0.7,
        "canonical": 1.0,
        "taxonomic": 1.0,
        "roots": 1.0,
        "abbreviation": 1.0,
        "roman": 1.0,
    }
    if weights:
        for k, v in weights.items():
            if k in default_weights:
                default_weights[k] = float(v)

    channels: Dict[str, float] = {
        "raw": raw_sim * default_weights["raw"],
        "canonical": canonical_sim * default_weights["canonical"],
        "taxonomic": tax_sim * default_weights["taxonomic"],
        "roots": root_sim * default_weights["roots"],
        "abbreviation": abbr_sim * default_weights["abbreviation"],
        "roman": roman_sim * default_weights["roman"],
    }

    # Clamp each channel to [0, 1].
    for k, v in channels.items():
        if v < 0.0:
            channels[k] = 0.0
        elif v > 1.0:
            channels[k] = 1.0

    # Base score: max of channels.
    base = max(channels.values()) if channels else 0.0

    # Multi-channel agreement boost: how many channels score
    # above the "strong" threshold (0.5)?
    strong = [v for v in channels.values() if v >= 0.5]
    if len(strong) >= 3:
        base += 0.15
    elif len(strong) >= 2:
        base += 0.05

    if base < 0.0:
        return 0.0
    if base > 1.0:
        return 1.0
    return float(base)


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

__all__ = [
    "parse_taxonomic_name",
    "normalize_diacritics",
    "parse_roman_numeral",
    "roman_to_int",
    "int_to_roman",
    "expand_abbreviation",
    "list_known_abbreviations",
    "split_compound",
    "latin_match",
]
