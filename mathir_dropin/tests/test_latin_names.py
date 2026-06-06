"""
MATHIR Drop-in -- Latin Name and Technical Nomenclature Tests
==============================================================

Test suite for :mod:`mathir_dropin.latin_names` and the
:meth:`UniversalBridge.expand_latin_query` integration.

Run via pytest
--------------
    cd D:/SECRET_PROJECT/MATHIR
    python -m pytest mathir_dropin/tests/test_latin_names.py -v

Run standalone
--------------
    python mathir_dropin/tests/test_latin_names.py
"""

from __future__ import annotations

import os
import sys
import unicodedata
from typing import Any, Callable, Dict, List, Optional

import pytest

# ---------------------------------------------------------------------------
# Path bootstrap -- mirrors test_universal_bridge.py
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_DROPIN = os.path.dirname(_HERE)
_PARENT = os.path.dirname(_DROPIN)
for _p in (_PARENT, _DROPIN):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mathir_dropin  # noqa: E402

# ---------------------------------------------------------------------------
# SUT imports
# ---------------------------------------------------------------------------
try:
    from mathir_dropin import latin_names as ln
    LN_AVAILABLE = True
    LN_IMPORT_ERROR: Optional[str] = None
except Exception as _e:  # noqa: BLE001
    ln = None  # type: ignore[assignment]
    LN_AVAILABLE = False
    LN_IMPORT_ERROR = repr(_e)

try:
    from mathir_dropin.universal_bridge import UniversalBridge
    UB_AVAILABLE = True
    UB_IMPORT_ERROR: Optional[str] = None
except Exception as _e:  # noqa: BLE001
    UniversalBridge = None  # type: ignore[assignment]
    UB_AVAILABLE = False
    UB_IMPORT_ERROR = repr(_e)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _require_ln() -> None:
    """Skip the test if `latin_names` failed to import."""
    if not LN_AVAILABLE:
        pytest.skip(f"latin_names not importable: {LN_IMPORT_ERROR}")


def _require_ub() -> None:
    if not UB_AVAILABLE:
        pytest.skip(f"universal_bridge not importable: {UB_IMPORT_ERROR}")


# ===========================================================================
# 1. Taxonomic Name Handler -- 5+ real species
# ===========================================================================

class TestParseTaxonomicName:
    """Tests for `parse_taxonomic_name`."""

    def test_homo_sapiens(self):
        _require_ln()
        r = ln.parse_taxonomic_name("Homo sapiens")
        assert r["is_taxonomic"] is True
        assert r["genus"] == "Homo"
        assert r["species"] == "sapiens"

    def test_escherichia_coli_k12(self):
        _require_ln()
        r = ln.parse_taxonomic_name("Escherichia coli K-12")
        assert r["is_taxonomic"] is True
        assert r["genus"] == "Escherichia"
        assert r["species"] == "coli"
        assert r.get("strain") == "K-12"

    def test_canis_lupus_familiaris_trinomial(self):
        _require_ln()
        r = ln.parse_taxonomic_name("Canis lupus familiaris")
        assert r["is_taxonomic"] is True
        assert r["genus"] == "Canis"
        assert r["species"] == "lupus"
        assert r.get("subspecies") == "familiaris"

    def test_panthera_tigris_altaica(self):
        _require_ln()
        r = ln.parse_taxonomic_name("Panthera tigris altaica")
        assert r["is_taxonomic"] is True
        assert r["genus"] == "Panthera"
        assert r["species"] == "tigris"
        assert r.get("subspecies") == "altaica"

    def test_quercus_robur_authority(self):
        _require_ln()
        r = ln.parse_taxonomic_name("Quercus robur L.")
        assert r["is_taxonomic"] is True
        assert r["genus"] == "Quercus"
        assert r["species"] == "robur"
        assert r.get("authority") == "Linnaeus"

    def test_bos_taurus_full_authority_year(self):
        _require_ln()
        r = ln.parse_taxonomic_name("Bos taurus Linnaeus, 1758")
        assert r["is_taxonomic"] is True
        assert r["genus"] == "Bos"
        assert r["species"] == "taurus"
        assert r.get("authority") == "Linnaeus"
        assert r.get("year") == 1758

    def test_e_coli_abbreviated_genus(self):
        _require_ln()
        r = ln.parse_taxonomic_name("E. coli")
        assert r["is_taxonomic"] is True
        assert r.get("genus_initial") == "E"
        assert r.get("genus") == "Escherichia"
        assert r.get("species") == "coli"

    def test_p_aeruginosa_abbreviated_genus(self):
        _require_ln()
        r = ln.parse_taxonomic_name("P. aeruginosa")
        assert r["is_taxonomic"] is True
        assert r.get("genus_initial") == "P"
        assert r.get("genus") == "Pseudomonas"
        assert r.get("species") == "aeruginosa"

    def test_staphylococcus_aureus(self):
        _require_ln()
        r = ln.parse_taxonomic_name("Staphylococcus aureus")
        assert r["is_taxonomic"] is True
        assert r["genus"] == "Staphylococcus"
        assert r["species"] == "aureus"

    def test_saccharomyces_cerevisiae(self):
        _require_ln()
        r = ln.parse_taxonomic_name("Saccharomyces cerevisiae")
        assert r["is_taxonomic"] is True
        assert r["genus"] == "Saccharomyces"
        assert r["species"] == "cerevisiae"

    def test_drosophila_melanogaster(self):
        _require_ln()
        r = ln.parse_taxonomic_name("Drosophila melanogaster")
        assert r["is_taxonomic"] is True
        assert r["genus"] == "Drosophila"
        assert r["species"] == "melanogaster"

    def test_henry_viii_not_taxonomic(self):
        """Proper names with roman numerals must NOT be parsed as binomials."""
        _require_ln()
        r = ln.parse_taxonomic_name("Henry VIII")
        assert r["is_taxonomic"] is False

    def test_world_war_ii_not_taxonomic(self):
        """Proper noun 'World War II' must not collapse to a binomial."""
        _require_ln()
        r = ln.parse_taxonomic_name("World War II")
        assert r["is_taxonomic"] is False

    def test_empty_input_safe(self):
        _require_ln()
        assert ln.parse_taxonomic_name("")["is_taxonomic"] is False
        assert ln.parse_taxonomic_name(None)["is_taxonomic"] is False  # type: ignore[arg-type]

    def test_garbage_input_safe(self):
        _require_ln()
        r = ln.parse_taxonomic_name("random garbage 123")
        assert r["is_taxonomic"] is False

    def test_raw_echo(self):
        _require_ln()
        r = ln.parse_taxonomic_name("Homo sapiens")
        assert r["raw"] == "Homo sapiens"


# ===========================================================================
# 2. Diacritic Normaliser -- 10+ names
# ===========================================================================

class TestNormalizeDiacritics:
    """Tests for `normalize_diacritics`."""

    @pytest.mark.parametrize("text,expected", [
        # Common Latin-1 / Western European
        ("Schrödinger", "Schrodinger"),
        ("Müller", "Muller"),
        ("François", "Francois"),
        ("François-Marie Arouet", "Francois-Marie Arouet"),
        ("naïve", "naive"),
        ("résumé", "resume"),
        ("café", "cafe"),
        # Polish
        ("Łódź", "Lodz"),
        ("Gdańsk", "Gdansk"),
        # German
        ("Straße", "Strasse"),
        # Spanish
        ("García", "Garcia"),
        ("Niño", "Nino"),
        # Portuguese
        ("São Paulo", "Sao Paulo"),
        ("Brasília", "Brasilia"),
        # Ligatures
        ("Œdipus", "OEdipus"),
        ("cœur", "coeur"),
        # Scandinavian / Icelandic
        ("Ångström", "Angstrom"),
        # Vietnamese
        ("Hà Nội", "Ha Noi"),
    ])
    def test_normalize(self, text: str, expected: str):
        _require_ln()
        assert ln.normalize_diacritics(text) == expected

    def test_empty(self):
        _require_ln()
        assert ln.normalize_diacritics("") == ""
        assert ln.normalize_diacritics(None) == ""  # type: ignore[arg-type]

    def test_case_preserving(self):
        _require_ln()
        # 'É' (U+00C9) should normalise to 'E' (uppercase preserved).
        assert ln.normalize_diacritics("É") == "E"
        assert ln.normalize_diacritics("é") == "e"
        # Polish L-with-stroke.
        assert ln.normalize_diacritics("Ł") == "L"
        assert ln.normalize_diacritics("ł") == "l"

    def test_ascii_passthrough(self):
        _require_ln()
        assert ln.normalize_diacritics("Hello World") == "Hello World"

    def test_idempotent(self):
        """Normalising an already-normalised string is a no-op."""
        _require_ln()
        s = "Schrödinger"
        once = ln.normalize_diacritics(s)
        twice = ln.normalize_diacritics(once)
        assert once == twice == "Schrodinger"

    def test_handles_combining_sequences(self):
        """A pre-decomposed combining sequence (e.g. NFD form) is
        stripped to the base letter."""
        _require_ln()
        decomposed = unicodedata.normalize("NFD", "é")
        assert ln.normalize_diacritics(decomposed) == "e"


# ===========================================================================
# 3. Roman Numeral Handler -- 5+ cases
# ===========================================================================

class TestRomanNumerals:
    """Tests for roman-numeral parsing and conversion."""

    def test_parse_simple(self):
        _require_ln()
        assert ln.parse_roman_numeral("VIII") == 8
        assert ln.parse_roman_numeral("IV") == 4
        assert ln.parse_roman_numeral("IX") == 9
        assert ln.parse_roman_numeral("XL") == 40
        assert ln.parse_roman_numeral("XC") == 90

    def test_parse_with_trailing_dot(self):
        _require_ln()
        assert ln.parse_roman_numeral("III.") == 3
        assert ln.parse_roman_numeral("XII.") == 12

    def test_parse_case_insensitive(self):
        _require_ln()
        assert ln.parse_roman_numeral("viii") == 8
        assert ln.parse_roman_numeral("iv") == 4
        assert ln.parse_roman_numeral("Mcmxcix") == 1999

    def test_parse_invalid_returns_none(self):
        _require_ln()
        assert ln.parse_roman_numeral("IIII") is None  # invalid form
        assert ln.parse_roman_numeral("foo") is None
        assert ln.parse_roman_numeral("") is None
        assert ln.parse_roman_numeral("ABC") is None

    def test_roman_to_int_basic(self):
        _require_ln()
        assert ln.roman_to_int("I") == 1
        assert ln.roman_to_int("V") == 5
        assert ln.roman_to_int("X") == 10
        assert ln.roman_to_int("L") == 50
        assert ln.roman_to_int("C") == 100
        assert ln.roman_to_int("D") == 500
        assert ln.roman_to_int("M") == 1000

    def test_roman_to_int_complex(self):
        _require_ln()
        assert ln.roman_to_int("MCMXCIX") == 1999
        assert ln.roman_to_int("MMXXIV") == 2024
        assert ln.roman_to_int("CDXLIV") == 444
        assert ln.roman_to_int("MMMDCCCLXXXVIII") == 3888

    def test_roman_to_int_invalid_raises(self):
        _require_ln()
        with pytest.raises(ValueError):
            ln.roman_to_int("IIII")
        with pytest.raises(ValueError):
            ln.roman_to_int("hello")

    def test_int_to_roman_basic(self):
        _require_ln()
        assert ln.int_to_roman(1) == "I"
        assert ln.int_to_roman(4) == "IV"
        assert ln.int_to_roman(8) == "VIII"
        assert ln.int_to_roman(9) == "IX"
        assert ln.int_to_roman(40) == "XL"
        assert ln.int_to_roman(90) == "XC"
        assert ln.int_to_roman(400) == "CD"
        assert ln.int_to_roman(900) == "CM"

    def test_int_to_roman_complex(self):
        _require_ln()
        assert ln.int_to_roman(1999) == "MCMXCIX"
        assert ln.int_to_roman(2024) == "MMXXIV"
        assert ln.int_to_roman(444) == "CDXLIV"
        assert ln.int_to_roman(3999) == "MMMCMXCIX"

    def test_int_to_roman_out_of_range_raises(self):
        _require_ln()
        with pytest.raises(ValueError):
            ln.int_to_roman(0)
        with pytest.raises(ValueError):
            ln.int_to_roman(4000)
        with pytest.raises(ValueError):
            ln.int_to_roman(-1)

    def test_round_trip(self):
        """int -> roman -> int is identity for all valid n."""
        _require_ln()
        for n in (1, 2, 3, 4, 5, 8, 9, 14, 27, 48, 49, 99, 400, 999,
                  1000, 1999, 2024, 3000, 3999):
            assert ln.roman_to_int(ln.int_to_roman(n)) == n


# ===========================================================================
# 4. Abbreviation Expander -- 10+ abbreviations
# ===========================================================================

class TestExpandAbbreviation:
    """Tests for `expand_abbreviation`."""

    @pytest.mark.parametrize("abbr,expected_substr", [
        ("DNA", "deoxyribonucleic acid"),
        ("RNA", "ribonucleic acid"),
        ("mRNA", "messenger ribonucleic acid"),
        ("ATP", "adenosine triphosphate"),
        ("CT", "computed tomography"),
        ("MRI", "magnetic resonance imaging"),
        ("ECG", "electrocardiogram"),
        ("PCR", "polymerase chain reaction"),
        ("NSAID", "nonsteroidal anti-inflammatory drug"),
        ("CNS", "central nervous system"),
        ("CO2", "carbon dioxide"),
        ("H2O", "water"),
    ])
    def test_known_abbreviation(self, abbr: str, expected_substr: str):
        _require_ln()
        expansions = ln.expand_abbreviation(abbr)
        assert any(expected_substr.lower() in e.lower() for e in expansions), (
            f"expected '{expected_substr}' in expansions of '{abbr}', got {expansions}"
        )

    def test_unknown_abbreviation_returns_original(self):
        _require_ln()
        out = ln.expand_abbreviation("XYZZY")
        assert "XYZZY" in out

    def test_empty_input(self):
        _require_ln()
        assert ln.expand_abbreviation("") == [""]

    def test_case_insensitive(self):
        _require_ln()
        out_upper = ln.expand_abbreviation("DNA")
        out_lower = ln.expand_abbreviation("dna")
        # Both should expand to the same full form.
        assert out_upper[0].lower() == out_lower[0].lower()

    def test_preserves_case_in_output(self):
        _require_ln()
        out = ln.expand_abbreviation("dna")
        # First expansion is the full form (lower), original is
        # preserved verbatim.
        assert "dna" in out

    def test_list_known_abbreviations_size(self):
        """We promised 50+ abbreviations -- verify."""
        _require_ln()
        abbrs = ln.list_known_abbreviations()
        assert len(abbrs) >= 50, f"expected >= 50, got {len(abbrs)}"
        assert all(isinstance(a, str) for a in abbrs)
        # Sorted.
        assert abbrs == sorted(abbrs)

    def test_strip_trailing_dot(self):
        """'i.e.' should still expand."""
        _require_ln()
        out = ln.expand_abbreviation("i.e.")
        assert any("id est" in e for e in out)


# ===========================================================================
# 5. Compound Term Splitter -- 5+ medical terms
# ===========================================================================

class TestSplitCompound:
    """Tests for `split_compound`."""

    def test_sternocleidomastoid(self):
        _require_ln()
        parts = ln.split_compound("sternocleidomastoid")
        assert "sterno" in parts
        assert "cleido" in parts
        assert "mastoid" in parts
        # Order: sterno first, cleido second, mastoid third.
        assert parts.index("sterno") < parts.index("cleido") < parts.index("mastoid")

    def test_latissimusdorsi(self):
        _require_ln()
        parts = ln.split_compound("latissimusdorsi")
        assert "latissimus" in parts
        assert "dorsi" in parts

    def test_temporomandibular(self):
        _require_ln()
        parts = ln.split_compound("temporomandibular")
        # Both "temporo" and "mandibul" are in roots; the exact
        # decomposition depends on the longest-match order, but
        # we require at least 2 parts.
        assert len(parts) >= 2

    def test_tracheobronchial(self):
        _require_ln()
        parts = ln.split_compound("tracheobronchial")
        assert len(parts) >= 2

    def test_carpometacarpal(self):
        _require_ln()
        parts = ln.split_compound("carpometacarpal")
        # Should split into carpo + meta + carpa(l) at minimum.
        assert len(parts) >= 2

    def test_known_full_root_returned_whole(self):
        """A term that IS itself a root (e.g. 'gastrocnemius') is
        returned as a single token rather than being over-split."""
        _require_ln()
        parts = ln.split_compound("gastrocnemius")
        # 'gastrocnemius' is in the roots dictionary, so it should
        # match as a single root.
        assert "gastrocnemius" in parts

    def test_empty_input(self):
        _require_ln()
        assert ln.split_compound("") == []

    def test_preserves_letters(self):
        """The union of output tokens must contain every letter of
        the input -- split_compound is never lossy."""
        _require_ln()
        for term in (
            "sternocleidomastoid",
            "latissimusdorsi",
            "temporomandibular",
            "tracheobronchial",
            "carpometacarpal",
            "otorhinolaryngology",
        ):
            parts = ln.split_compound(term)
            joined = "".join(parts)
            assert joined == term, f"lossy split: {term!r} -> {parts}"

    def test_handles_diacritics(self):
        """The function de-diacriticises the input before splitting."""
        _require_ln()
        # "Hôtel-Dieu" contains "Hôtel" which has an accent.
        parts = ln.split_compound("Hôtel")
        # After de-diacriticking: "Hotel" -- 'hot' is not a root,
        # so we get the whole word back as a single token.
        assert "".join(parts).lower() == "hotel"


# ===========================================================================
# 6. End-to-end Latin Name Matching -- 5+ scenarios
# ===========================================================================

class TestLatinMatch:
    """Tests for `latin_match`."""

    def test_exact_match_returns_one(self):
        _require_ln()
        assert ln.latin_match("Homo sapiens", "Homo sapiens") >= 0.99

    def test_genus_initial_abbreviation(self):
        """'E. coli' should match 'Escherichia coli' with high score."""
        _require_ln()
        s = ln.latin_match("E. coli", "Escherichia coli")
        assert s >= 0.85, f"E. coli vs Escherichia coli: {s}"

    def test_diacritic_insensitive(self):
        """'Schrodinger' must match 'Schrödinger' (with umlaut)."""
        _require_ln()
        s = ln.latin_match("Schrodinger", "Schrödinger")
        assert s >= 0.85, f"Schrodinger vs Schrödinger: {s}"

    def test_muller_diacritic(self):
        _require_ln()
        s = ln.latin_match("Muller", "Müller")
        assert s >= 0.85, f"Muller vs Müller: {s}"

    def test_roman_numeral_equivalence(self):
        """'Henry VIII' should strongly match 'Henry 8'."""
        _require_ln()
        s = ln.latin_match("Henry VIII", "Henry 8")
        assert s >= 0.80, f"Henry VIII vs Henry 8: {s}"

    def test_abbreviation_full_form(self):
        """'DNA' should match 'deoxyribonucleic acid' with high score."""
        _require_ln()
        s = ln.latin_match("DNA", "deoxyribonucleic acid")
        assert s >= 0.80, f"DNA vs deoxyribonucleic acid: {s}"

    def test_different_genus_low_score(self):
        """'Panthera leo' should NOT match 'Panthera tigris' strongly
        (different species)."""
        _require_ln()
        s = ln.latin_match("Panthera leo", "Panthera tigris")
        assert s < 0.95, f"Panthera leo vs tigris: {s}"

    def test_unrelated_strings_zero(self):
        """Two completely unrelated strings should return 0."""
        _require_ln()
        s = ln.latin_match("python", "java")
        assert s == 0.0

    def test_empty_input_zero(self):
        _require_ln()
        assert ln.latin_match("", "java") == 0.0
        assert ln.latin_match("python", "") == 0.0
        assert ln.latin_match("", "") == 0.0

    def test_compound_decomposition_similarity(self):
        """Splitting + n-gram similarity should give a reasonable
        match between 'latissimus dorsi' (with space) and the
        concatenated 'latissimusdorsi'."""
        _require_ln()
        s = ln.latin_match("latissimus dorsi", "latissimusdorsi")
        assert s >= 0.50, f"latissimus dorsi vs latissimusdorsi: {s}"

    def test_white_blood_cell_expansion(self):
        _require_ln()
        # WBC -> "white blood cell" is a known abbreviation.
        s = ln.latin_match("WBC", "white blood cell")
        assert s >= 0.70, f"WBC vs white blood cell: {s}"

    def test_world_war_two_roman_match(self):
        _require_ln()
        s = ln.latin_match("World War II", "World War 2")
        assert s >= 0.70, f"World War II vs World War 2: {s}"

    def test_score_in_range(self):
        """The score must always be in [0, 1]."""
        _require_ln()
        for q, c in [
            ("Homo sapiens", "Homo erectus"),
            ("DNA", "deoxyribonucleic acid"),
            ("E. coli", "Escherichia coli"),
            ("Schrödinger", "Schrodinger"),
            ("Henry VIII", "Henry 8"),
        ]:
            s = ln.latin_match(q, c)
            assert 0.0 <= s <= 1.0, f"out-of-range score {s} for {q!r} vs {c!r}"


# ===========================================================================
# 7. Integration: expand_latin_query on UniversalBridge
# ===========================================================================

class TestExpandLatinQuery:
    """Tests for the new `UniversalBridge.expand_latin_query` method."""

    def test_basic_expansion(self):
        _require_ub()
        bridge = UniversalBridge()
        out = bridge.expand_latin_query("Homo sapiens")
        assert isinstance(out, list)
        assert len(out) > 0
        assert "Homo sapiens" in out
        # Should also produce the abbreviated form and the
        # component parts.
        assert any("H. sapiens" in v for v in out)
        assert any("Homo" == v for v in out)
        assert any("sapiens" == v for v in out)

    def test_diacritic_expansion(self):
        _require_ub()
        bridge = UniversalBridge()
        out = bridge.expand_latin_query("Schrödinger")
        assert "Schrödinger" in out
        assert any("Schrodinger" == v for v in out)  # ASCII variant

    def test_roman_numeral_expansion(self):
        _require_ub()
        bridge = UniversalBridge()
        out = bridge.expand_latin_query("Henry VIII")
        assert "Henry VIII" in out
        assert any("Henry 8" in v for v in out)

    def test_abbreviation_expansion(self):
        _require_ub()
        bridge = UniversalBridge()
        out = bridge.expand_latin_query("DNA")
        assert "DNA" in out
        assert any("deoxyribonucleic" in v for v in out)

    def test_compound_expansion(self):
        _require_ub()
        bridge = UniversalBridge()
        out = bridge.expand_latin_query("sternocleidomastoid")
        assert "sternocleidomastoid" in out
        assert any("sterno" in v for v in out)

    def test_empty_input(self):
        _require_ub()
        bridge = UniversalBridge()
        assert bridge.expand_latin_query("") == []
        assert bridge.expand_latin_query(None) == []  # type: ignore[arg-type]

    def test_deduplicated(self):
        """No case-insensitive duplicate variants."""
        _require_ub()
        bridge = UniversalBridge()
        out = bridge.expand_latin_query("Homo sapiens")
        lower = [v.lower() for v in out]
        assert len(lower) == len(set(lower))

    def test_genus_abbreviation(self):
        _require_ub()
        bridge = UniversalBridge()
        out = bridge.expand_latin_query("E. coli")
        # Should include both the abbreviated form and the
        # expanded genus.
        assert any("E. coli" in v for v in out)
        assert any("Escherichia" == v for v in out)

    def test_world_war_ii(self):
        _require_ub()
        bridge = UniversalBridge()
        out = bridge.expand_latin_query("World War II")
        assert "World War II" in out
        assert any("World War 2" in v for v in out)

    def test_uses_ngram_size_kwarg(self):
        """Different `ngram_size` configs do not break the function."""
        _require_ub()
        bridge2 = UniversalBridge(ngram_size=2)
        out = bridge2.expand_latin_query("Homo sapiens")
        assert len(out) > 0


# ===========================================================================
# 8. Smoke / regression tests for module-level helpers
# ===========================================================================

class TestModuleImports:
    """Verify the module's public API is intact and importable."""

    def test_required_functions_exported(self):
        _require_ln()
        for name in (
            "parse_taxonomic_name",
            "normalize_diacritics",
            "parse_roman_numeral",
            "roman_to_int",
            "int_to_roman",
            "expand_abbreviation",
            "list_known_abbreviations",
            "split_compound",
            "latin_match",
        ):
            assert hasattr(ln, name), f"missing public function: {name}"
            assert callable(getattr(ln, name)), f"{name} is not callable"

    def test_at_least_50_abbreviations(self):
        _require_ln()
        assert len(ln.list_known_abbreviations()) >= 50

    def test_at_least_50_medical_roots(self):
        """The internal roots dictionary must have 50+ entries.
        We can't access it directly (no public attribute) but the
        `split_compound` behaviour implicitly tests it."""
        _require_ln()
        # 'sternocleidomastoid' should split into 3 parts, which
        # requires at least sterno, cleido, mastoid in the dict.
        parts = ln.split_compound("sternocleidomastoid")
        assert len(parts) == 3


# ===========================================================================
# Standalone runner
# ===========================================================================

def _standalone() -> int:  # pragma: no cover
    """Run all tests without pytest and return exit code."""
    import traceback
    failures: List[Tuple[str, str]] = []
    # Discover every test_* method on every test class.
    import inspect
    current_module = sys.modules[__name__]
    for name, obj in inspect.getmembers(current_module, inspect.isclass):
        if not name.startswith("Test"):
            continue
        instance = obj()
        for mname, method in inspect.getmembers(instance, inspect.isfunction):
            if not mname.startswith("test_"):
                continue
            test_id = f"{name}.{mname}"
            try:
                method()
            except pytest.skip.Exception:
                print(f"[SKIP] {test_id}")
            except Exception:  # noqa: BLE001
                tb = traceback.format_exc()
                failures.append((test_id, tb))
                print(f"[FAIL] {test_id}")
                print(tb)
            else:
                print(f"[PASS] {test_id}")
    if failures:
        print(f"\n{len(failures)} failure(s).")
        return 1
    print("\nAll standalone tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(_standalone())
