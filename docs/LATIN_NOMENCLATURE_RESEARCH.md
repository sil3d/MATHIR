# Research Report — Latin Names, Scientific Nomenclature & Technical Terms

**Compiled by:** @background-researcher
**Date:** 2026-06-06
**Project:** MATHIR (universal_bridge / `latin_names.py`)
**Scope:** Algorithm patterns for matching/normalizing Latinate scientific, technical, and proper-noun terminology in cross-lingual IR.

---

## TL;DR

Latin/scientific/technical terms obey a small set of **deep, regular structural rules** (binomial n-grams, INN stems, Bayer letter+genitive, diacritic classes, Roman-numeral contexts). Algorithms that fail to recognize these patterns lose precision on the most valuable technical vocabulary. The **5-7 patterns an algorithm MUST handle** are summarized at the end.

---

## 1. Taxonomic Nomenclature (biology)

### 1.1 Standards & Authorities

- **ICZN** — International Code of Zoological Nomenclature (animals). Governs "binomial nomenclature" since Linnaeus, *Systema Naturae* 10th ed., 1758.
- **ICN** — International Code of Nomenclature for algae, fungi, and plants (formerly ICBN). Uses "binomial nomenclature" with the "subsp." abbreviation (vs. ICZN's "ssp.").
- Both are LATIN, gender-disagreeing nouns/adjectives; the genus name is a **Latinized noun** (always capitalized, treated as a singular proper noun), the species epithet is **lowercase**.

### 1.2 Core structure (formal grammar)

```
TaxonName      := Genus Species [Author, Year]
Genus          := Capitalized Latin noun (1 word; may be hyphenated)
Species        := lowercase Latin epithet (1 word, may be 2+ words if hyphenated)
                | "sp." | "spp." | "subsp." <epithet> | "var." <epithet>
Author         := Surname[, Year][(comb. nov.)] | "et al."
Year           := 4-digit year (1758+)
```

EBNF-ish: `Binomial = (Upper Word) (lower Word)  ;  Trinomial = Binomial (lower Word)  ;  Authority = (Upper) ", " (4-digit)`

### 1.3 Author citation rules (ICZN)

- After first use, the **genus is abbreviated to its initial**: *Homo sapiens* → *H. sapiens*.
- Parens around author = species was moved from original genus:
  - *Vanessa atalanta* (Linnaeus, 1758) — Linnaeus originally placed it in *Papilio*.
  - *Balaena mysticetus* Linnaeus, 1758 — bowhead whale; no parens = Linnaeus's original genus.
- Authors with two names are NOT typically abbreviated (per ICZN).
- Authority may be omitted in informal contexts.
- Abbreviation conventions: `&` for "and", `ex` for "validly published by", `sensu` (NOT italicized) for "as used by".

### 1.4 Standard abbreviations (used inside the canonical name)

| Abbrev | Meaning | Notes |
|--------|---------|-------|
| `sp.` | one unspecified species | NOT italicized |
| `spp.` | several unspecified species | NOT italicized |
| `ssp.` (zoo) / `subsp.` (bot) | subspecies | followed by epithet |
| `sspp.` / `subspp.` | multiple subspecies | |
| `var.` | variety (botany) | |
| `f.` / `forma` | form (botany) | |
| `auct.` | "of authors" (misapplication) | |
| `comb. nov.` | combinatio nova (new combination) | |
| `nom. nov.` | nomen novum (replacement name) | |
| `nom. nud.` | nomen nudum (unavailable) | |
| `cf.` | confer (compare with) | precedes species |

### 1.5 12+ Real Examples (with structure)

| # | Full name | Type | Notes |
|---|-----------|------|-------|
| 1 | *Homo sapiens* Linnaeus, 1758 | Binomial + authority | "wise human" — canonical human |
| 2 | *Homo sapiens sapiens* | Trinomial (subspecies) | extinct human subspecies |
| 3 | *Escherichia coli* | Binomial (bacteria) | shortened to *E. coli* in popular use |
| 4 | *Panthera leo* (Linnaeus, 1758) | Binomial; leo = noun in apposition | lion |
| 5 | *Panthera tigris* | Binomial | tiger |
| 6 | *Canis lupus familiaris* | Trinomial (subspecies of *C. lupus*) | domestic dog |
| 7 | *Canis lupus* Linnaeus, 1758 | Binomial + authority | gray wolf |
| 8 | *Tursiops truncatus* | Binomial | bottlenose dolphin |
| 9 | *Balaena mysticetus* Linnaeus, 1758 | Binomial + authority (no parens) | bowhead whale |
| 10 | *Ailuropoda melanoleuca* | Binomial | giant panda ("black-and-white cat-foot") |
| 11 | *Tyrannosaurus rex* | Binomial | popular shortened to *T. rex* |
| 12 | *Boa constrictor* Linnaeus, 1758 | Binomial | first part noun, second adjective |
| 13 | *Canis* sp. | Genus + "sp." | unspecified species of *Canis* |
| 14 | *Canis* spp. | Genus + "spp." | multiple species |
| 15 | *Escherichia coli* O157:H7 | Binomial + serotype (post-ICZN extension) | pathogenic strain |
| 16 | *Magnolia hodgsonii* | Binomial, genitive of "Hodgson" | "−ii" = male commemorated |
| 17 | *Anthus hodgsoni* | Binomial, genitive of "Hodgson" | "−i" (also male) — variant declension |
| 18 | *Quercus robur* subsp. *robur* | Trinomial with explicit subsp. | |

### 1.6 Edge cases & common errors

- **Case-mixing** is the #1 matching bug. *Homo sapiens* must match *HOMO SAPIENS*, *homo sapiens*, *H. sapiens*.
- The genus-initial abbreviation is **only valid AFTER a first full mention** in the same document; some search engines must therefore expand *E. coli* → *Escherichia coli* (E. coli is so widespread that it's accepted as a standalone common name).
- The epithet can be an **adjective agreeing with the genus** (*Panthera leo* uses "leo" as a noun in apposition, so no agreement needed) OR a **noun in genitive** (*coli* = "of the colon"; *hodgsonii* = "of Hodgson") OR a **noun in apposition** (*Panthera leo*).
- Hyphenated species epithets exist (e.g. *Lactobacillus kefiranofaciens* — single word) and rare multi-word epithets (deprecated by ICZN but appear historically).
- Strain identifiers (O157:H7, K-12) are appended with no italicization.

### 1.7 Algorithm implications (for MATHIR)

- Recognize a **2- or 3-token pattern** `[Capitalized] [lowercase] ( [lowercase] )?` where token boundaries may be the same string used as genus abbreviation later.
- A genus can be a **3-letter abbreviation** (e.g., *E. coli*, *T. rex*). Algorithm should expand these.
- Authority strings must be **optional and separable** from the canonical name with regex `\s*\([^)]*\d{4}\)|\s+\w+,?\s*\d{4}`.
- Subspecies/variety qualifiers (`subsp.`, `var.`, `f.`) followed by another token are **embedded** in the canonical name.

---

## 2. Anatomical Terminology (Terminologia Anatomica)

### 2.1 Standards

- **Terminologia Anatomica (TA)** — current standard, Federative Committee on Anatomical Terminology (FCAT) / IFAA, 1998. ~7,500 gross anatomy terms.
- **Terminologia Histologica** — histology. **Terminologia Neuroanatomica** — nervous system (merged 2016). **Terminologia Embryologica** — embryology.
- All terms are **Latin, listed in TA as official (no English equivalent)**.
- Supersedes *Nomina Anatomica* (1955-1985) and *BNA* (*Basle Nomina Anatomica*, 1895).

### 2.2 Core structure

A Latin anatomical term is a **concatenated noun phrase** that follows the template:

```
Term  := [Modifier]* Root [Modifier]*
```

Concrete templates (all in Latin):
- `[Noun] [adjective]` — *cavum nasi* (nasal cavity)
- `[genitive/possessive] [noun]` — *musculus brachii* (muscle of the arm)
- `[compound noun]` — *sternocleidomastoideus* (sterno + cleido + mastoideus)
- `[adjective] [noun in genitive]` — *arteria iliaca communis* (common iliac artery)

The order is typically **descriptive-modifier-then-anatomy-noun**, but sometimes the noun comes first as a heading:
- *musculus sternocleidomastoideus* (muscle, sternocleidomastoid) — noun first, then adjective
- *musculus biceps brachii* (biceps brachii muscle) — noun, then adjective phrase

### 2.3 Word formation rules (regular morphology)

| Element | Latin | Meaning | Example |
|---------|-------|---------|---------|
| structure | *musculus, arteria, vena, nervus, os, ligamentum, cartilago, glandula* | the body part class | *musculus biceps brachii* |
| shape | *-oideus* | resembling | *mast-oideus* (resembling a breast) |
| location | *anterior, posterior, medialis, lateralis, superior, inferior, proximalis, distalis* | relative position | *tibialis anterior* |
| size | *major, minor, maximus, minimus, longus, brevis* | comparative/superlative | *gluteus maximus* |
| number | *bi-, tri-, quadri-* | number of parts | *biceps, triceps* |
| direction | *-ad* (toward) | | *cephal-ad* (headward) |
| relation | *inter-, supra-, infra-, sub-* | | *intercostalis* (between ribs) |
| inflammation | *-itis* | (clinical) | *hepatitis* |
| study of | *-ologia* | (clinical) | *cardiologia* |

The same root can form noun/adjective by suffix:
- *cavum* (noun: cavity) / *cavus* (adj: hollow) / *cavitas* (noun: cavitation)
- *gaster* (Greek: stomach) / *gastricus* (adj)

### 2.4 Eponyms (named after people)

In clinical/anatomical practice, eponyms (named after discoverers) are common **ALONGSIDE** the Latin term. Both are valid; modern TA prefers Latin.

| Latin TA term | Eponym (English) | Anatomist / Physician |
|---------------|------------------|----------------------|
| *area parolfactoria* (Broca's area) | Broca's area | Paul Broca, 1861 |
| *gyrus frontalis inferior* pars opercularis | Broca's area (modern) | — |
| *morbus Alzheimerianus* | Alzheimer's disease | Alois Alzheimer, 1906 |
| *circulus arteriosus cerebri* | Circle of Willis | Thomas Willis, 1664 |
| *ductus thoracicus* | (no common eponym) | — |
| *tuba uterina* | Fallopian tube | Gabriele Falloppio, 1561 |
| *aponeurosis bicipitalis* | lacertus fibrosus (no eponym) | — |
| *musculus trapezius* | (no eponym) | — |
| *nodus sinuatrialis* | SA node / Keith-Flack node | Arthur Keith, Martin Flack, 1906 |
| *ligamentum cruciatum anterius* | ACL (no eponym) | — |
| *medulla spinalis* | spinal cord | — |
| *ventriculus sinister* | left ventricle | — |

### 2.5 10+ Real Examples (with structure breakdown)

| # | Official Latin (TA) | Common English | Structure |
|---|---------------------|----------------|-----------|
| 1 | *musculus sternocleidomastoideus* | sternocleidomastoid muscle | muscul-[us] stern-o-cleid-o-mast-oide-[us] (3-bone compound adj) |
| 2 | *musculus biceps brachii* | biceps brachii (two-headed muscle of arm) | muscul-[us] bi-ceps brach-i-i |
| 3 | *musculus triceps surae* | triceps surae (calf) | muscul-[us] tri-ceps sur-ae |
| 4 | *musculus gastrocnemius* | gastrocnemius (calf) | gastr-o-cnemi-[us] (belly + leg) |
| 5 | *musculus latissimus dorsi* | latissimus dorsi (broad back) | muscul-[us] lat-issim-[us] dors-i |
| 6 | *musculus gluteus maximus* | gluteus maximus | muscul-[us] glut-[eus] maxim-[us] |
| 7 | *musculus trapezius* | trapezius | muscul-[us] trapez-[ius] |
| 8 | *musculus deltoideus* | deltoid | muscul-[us] delt-oide-[us] |
| 9 | *musculus tibialis anterior* | tibialis anterior | muscul-[us] tibi-al-[is] anterior |
| 10 | *musculus rectus abdominis* | rectus abdominis (abs) | muscul-[us] rect-[us] abdominis |
| 11 | *musculus pectoralis major* | pectoralis major | muscul-[us] pector-al-[is] major |
| 12 | *arteria carotis communis* | common carotid artery | arteri-a carot-[is] commun-[is] |
| 13 | *vena cava superior* | superior vena cava | ven-a cav-a superior |
| 14 | *nervus vagus* | vagus nerve | nerv-[us] vag-[us] |
| 15 | *os femoris* / *femur* | femur (thigh bone) | os femor-[is] |
| 16 | *ligamentum cruciatum anterius* | anterior cruciate ligament (ACL) | ligament-[um] cruciat-[um] anter-[ius] |

### 2.6 Common English shortenings vs. Latin TA

This is a major source of mismatch:
- "sternocleidomastoid" (English) ↔ "musculus sternocleidomastoideus" (TA) — drop *musculus*, change final **−us → −oid** (English uses -oid, Latin uses -oideus)
- "gastrocnemius" = same form, no shortening
- "deltoid" ↔ "musculus deltoideus" — drop *musculus*, change **-eus → -oid** again
- "lat dorsi" (clinical slang) ↔ "musculus latissimus dorsi" — full shortening
- "quad" / "quads" (colloquial) ↔ "musculus quadriceps femoris"

### 2.7 Algorithm implications

- Recognize **the `musculus [latin adj]` pattern** and collapse to the Latin adj (since the adj carries the meaning).
- Recognize **the **-us → -oid** English-shortening rule** (and the inverse: -oideus → -oid).
- Eponyms are **1+ capitalized word + 's/''** and should be **indexed to the Latin TA term** in a synonym table.
- "Anterior/posterior/medial/lateral" are **direction terms that may prefix OR follow the noun** in English but always follow in Latin.

---

## 3. Pharmaceutical Names (INN system)

### 3.1 Standards

- **INN** = International Nonproprietary Name, selected by **WHO** (mandate from 1953).
- Published in **English, Latin, French, Russian, Spanish, Arabic, Chinese** — same stem across all.
- **rINN** = recommended INN; **pINN** = proposed INN; **INNM** = modified INN (for salts/esters).
- National equivalents: **BAN** (British), **USAN** (US), **JAN** (Japan), **DCF** (France DCF), **DCIT** (Italy). These have largely converged with INN.

### 3.2 Stem-based naming

**A stem is a syllable(s) signaling pharmacological class or chemistry.** The full INN system has **> 400 stems** published in WHO's *Stem Book* (2024 update).

The key rule: **stem appears at the end (suffix) usually, sometimes at the beginning (prefix).**

| Stem | Class | Examples |
|------|-------|----------|
| `-pril` | ACE inhibitors | captopril, enalapril, lisinopril |
| `-sartan` | Angiotensin II receptor antagonists (ARBs) | losartan, valsartan |
| `-statin` | HMG-CoA reductase inhibitors (cholesterol) | atorvastatin, simvastatin, rosuvastatin |
| `-olol` | Beta-blockers | propranolol, atenolol, metoprolol |
| `-dipine` | Dihydropyridine calcium channel blockers | amlodipine, nifedipine, felodipine |
| `-mab` | Monoclonal antibodies | infliximab, adalimumab, rituximab |
| `-nib` | Small-molecule tyrosine kinase inhibitors | imatinib, erlotinib, dasatinib |
| `-tide` | Peptides | octreotide, liraglutide |
| `-vir` | Antivirals | remdesivir, ritonavir, acyclovir |
| `-cillin` | Penicillin antibiotics | amoxicillin, ampicillin, oxacillin |
| `cef-` / `ceph-` | Cephalosporin antibiotics | cefalexin, ceftriaxone, cefazolin |
| `-mycin` | Aminoglycoside antibiotics | streptomycin, gentamicin, vancomycin |
| `-floxacin` | Fluoroquinolones | ciprofloxacin, levofloxacin, moxifloxacin |
| `-azole` | Antifungals | fluconazole, ketoconazole, metronidazole |
| `-azepam` | Benzodiazepines | lorazepam, diazepam, oxazepam |
| `-prazole` | Proton pump inhibitors (PPIs) | omeprazole, lansoprazole, pantoprazole |
| `-tidine` | H2 receptor antagonists | ranitidine, cimetidine, famotidine |
| `-parin` | Heparin derivatives | enoxaparin, dalteparin |
| `-ase` | Enzymes | alteplase, streptokinase, pancrelipase |
| `-ast` | Anti-asthmatics / anti-allergics (some) | zafirlukast, montelukast |

### 3.3 Chemical name ↔ INN ↔ brand name

| IUPAC chemical | INN (generic) | Common brand | Stem class |
|----------------|---------------|--------------|------------|
| 2-acetoxybenzoic acid | acetylsalicylic acid | Aspirin | (-in/-ine family historically) |
| (S)-N-(2,6-dimethylphenyl)-1-methyl-2-piperidinecarboxamide | bupivacaine | Marcaine, Sensorcaine | -caine (local anesthetics) |
| N-(4-hydroxyphenyl)acetamide | paracetamol (INN) / acetaminophen (USAN) | Tylenol, Panadol | (no specific class stem) |
| (RS)-2-(4-(2-methylpropyl)phenyl)propanoic acid | ibuprofen | Advil, Nurofen | -profen (NSAIDs/arylpropionic acids) |
| 7-chloro-1,3-dihydro-1-methyl-5-phenyl-2H-1,4-benzodiazepin-2-one | diazepam | Valium | -azepam |
| methyl (E)-2-[[3-(1-piperidinylmethyl)phenoxy]methyl]benzeneacetate | loratadine | Claritin | -tadine (antihistamines) |
| 2,2-dimethylpropionic acid, 2,6-dimethyl-4-(2-nitrophenyl)-3,5-pyridinedicarboxylic acid 1,4-dihydropyridine ester | nifedipine | Adalat, Procardia | -dipine |
| 4-amino-N-(5-methyl-3-isoxazolyl)benzenesulfonamide | sulfamethoxazole | (in Bactrim) | sulfa- (sulfonamides) |
| (2S,3R,4R,5S,6R)-2-{[(2R,3S,4R,5R)-3,4-dihydroxy-2-(hydroxymethyl) tetrahydro-2H-pyran-5-yl]oxy}-6-(hydroxymethyl)tetrahydro-2H-pyran-3,4,5-triol | — (no INN) | — | (trivial names only) |
| 4-{[4-(4-chlorophenyl)-2-pyridinyl]methoxy}-4-oxobutanoic acid | — (no INN) | — | (chemistry only) |

### 3.4 INN spelling regularization (predictable rules)

Per WHO, INN follows a **phonemic orthography**:
- `ph` → `f` (*amfetamine* not *amphetamine*)
- `th` → `t` (*levmetamfetamine* not *levomethamphetamine*)
- `ae`, `oe` → `e` (*cefepime* not *cephaepime*)
- `y` → `i` (*furosemide* not *furosemide... wait — furosemide already follows this; *sodium picosulfate* not *picosulphate*)
- Avoid `h` and `k` if possible

This is the **#1 reason** drug names look "off" to non-specialists — *amfetamine* looks like a typo of *amphetamine*.

### 3.5 10+ Stem-Based Examples

1. **atorvastatin** → `-statin` (HMG-CoA reductase inhibitor) — Lipitor
2. **enalapril** → `-pril` (ACE inhibitor) — Vasotec
3. **losartan** → `-sartan` (ARB) — Cozaar
4. **amlodipine** → `-dipine` (CCB) — Norvasc
5. **adalimumab** → `-mab` (mAb; **-li-**-mab = immunomodulator) — Humira
6. **imatinib** → `-nib` (TKI) — Gleevec
7. **remdesivir** → `-vir` (antiviral) — Veklury
8. **omeprazole** → `-prazole` (PPI) — Prilosec
9. **lorazepam** → `-azepam` (benzodiazepine) — Ativan
10. **ciprofloxacin** → `-floxacin` (fluoroquinolone) — Cipro
11. **metoprolol** → `-olol` (β-blocker) — Lopressor/Toprol
12. **rituximab** → `-mab` (chimeric anti-CD20 mAb) — Rituxan
13. **budesonide** → `-ide` (steroid) — Pulmicort
14. **aciclovir / acyclovir** → `-vir` (antiviral) — Zovirax
15. **liraglutide** → `-tide` (GLP-1 peptide) — Victoza

### 3.6 Algorithm implications

- **Index INN stems as separate tokens** so `*-statin` matches all statins, etc.
- The same INN can be written in many surface forms:
  - *paracetamol* (INN/BAN) / *acetaminophen* (USAN) / *paracetamolum* (Latin) / *paracétamol* (FR) / парацетамол (RU)
  - The **first ~5 letters** are usually stable.
- Brand names are **trademarked**; never stem-match them. Index them only as synonyms.
- INN "modified" for salt: `<base> <salt>` (e.g., *oxacillin sodium*) — space-separated.

---

## 4. Legal Latin Phrases

### 4.1 Standards

- Latin legal terminology derives from **Classical and Medieval Latin**, surviving in common-law, civil-law, and ecclesiastical-law systems.
- Always **italicized** in legal writing (since non-English).
- Often **abbreviated**: e.g., *infra*, *supra*, *ibid.*, *op. cit.*, *q.v.*, *e.g.*, *i.e.*, *cf.*, *et seq.*, *et al.*
- In pleading/case names: capitalized (e.g., *In re Gault*, *Marbury v. Madison*).

### 4.2 Common Latin legal phrases (12+)

| # | Phrase | Pronunciation guide | Meaning | Example use |
|---|--------|---------------------|---------|-------------|
| 1 | **habeas corpus** | HAY-bee-uss KOR-puss | "you shall have the body" — writ demanding a prisoner be brought before a court to determine legality of detention | *Writ of habeas corpus* filed by detainee. |
| 2 | **pro bono** | proh BOH-noh | "for the public good" — done without charge, typically legal work | *The firm took the case pro bono.* |
| 3 | **pro bono publico** | proh BOH-noh PUB-li-koh | same as pro bono, more emphatic | *Work done pro bono publico.* |
| 4 | **ex parte** | ex PAR-tay | "from one side" — application made by one party without notice to the other | *Ex parte hearing* (no opposition present). |
| 5 | **amicus curiae** | ah-MEE-kuss KYOOR-ee-eye | "friend of the court" — non-party who offers expertise | *The ACLU filed an amicus brief.* |
| 6 | **prima facie** | PRY-mah FAY-shee(-ay) | "at first sight" — evidence sufficient on its face | *Prima facie case of negligence.* |
| 7 | **per se** | per SAY | "by/in itself" | *Not per se illegal.* |
| 8 | **mens rea** | menz RAY-ah | "guilty mind" — intent to commit a crime | *Theft requires mens rea.* |
| 9 | **actus reus** | AK-tuss RAY-uss | "guilty act" — physical component of a crime | *Both actus reus and mens rea required.* |
| 10 | **in absentia** | in ab-SEN-shee-ah | "in absence" — done while one party is not present | *Trial in absentia* (defendant not present). |
| 11 | **in re** | in RAY | "in the matter of" — used when no adversary | *In re Smith Estate.* |
| 12 | **in rem** | in REM | "against a thing" — jurisdiction over property, not person | *In rem forfeiture of the vessel.* |
| 13 | **in personam** | in per-SOH-nam | "against a person" — personal jurisdiction | *Judgment in personam.* |
| 14 | **corpus delicti** | KOR-puss dee-LIK-tee | "body of the crime" — the fact that a crime occurred | *Corpus delicti must be established.* |
| 15 | **corpus juris** | KOR-puss JOOR-iss | "body of law" — a comprehensive legal code | *Corpus Juris Civilis* (Justinian). |
| 16 | **duces tecum** | DOO-sess TAY-kum | "bring with you" — subpoena to produce documents | *Subpoena duces tecum.* |
| 17 | **sub judice** | sub YOO-di-kay | "under a judge" — before a court, not for public discussion | *The matter is sub judice.* |
| 18 | **subpoena** | sub-PEE-nah | "under penalty" — summons | *Subpoena ad testificandum* (to testify). |
| 19 | **quid pro quo** | kwid proh KWOH | "something for something" | *Quid pro quo sexual harassment.* |
| 20 | **ad hoc** | ad HOK | "for this" — improvised, special purpose | *Ad hoc committee.* |
| 21 | **ex post facto** | ex pohst FAK-toh | "after the fact" — retroactive (esp. criminal law) | *Ex post facto law is unconstitutional in US.* |
| 22 | **alibi** | AL-ih-bye | "elsewhere" — defense that defendant was elsewhere | *Alibi defense.* |
| 23 | **res ipsa loquitur** | rez IP-sah LOK-wi-tur | "the thing speaks for itself" — negligence inferred | *Res ipsa loquitur doctrine.* |
| 24 | **stare decisis** | STAR-ay de-SIGH-sis | "to stand by decided matters" — precedent | *Binding under stare decisis.* |
| 25 | **voir dire** | vwar DEER | "to speak the truth" — jury selection / pretrial examination | *Voir dire of prospective jurors.* |
| 26 | **malum in se** | MAH-lum in SAY | "wrong in itself" — inherently evil act | *Murder is malum in se.* |
| 27 | **malum prohibitum** | MAH-lum proh-HIB-i-tum | "wrong because prohibited" — act illegal only by statute | *Drug possession is malum prohibitum.* |
| 28 | **bona fide** | BOH-nah FY-dee | "in good faith" | *Bona fide purchaser.* |
| 29 | **inter alia** | in-ter AH-lee-ah | "among other things" | *The plaintiff claims, inter alia, fraud.* |
| 30 | **a fortiori** | ah for-shee-OR-ee | "from stronger [reason]" — even more so | *If A, then a fortiori B.* |

### 4.3 Use in legal documents

- Italicization is **mandatory** in Bluebook / OSCOLA / most jurisdictions.
- Cap when **first word of a citation** (e.g., *In re Smith*); lowercase otherwise.
- Common abbreviations: *infra* (below/later), *supra* (above/earlier), *ibid.* (same source), *op. cit.* (work cited), *q.v.* (which see), *e.g.* (for example), *i.e.* (that is), *cf.* (compare), *et seq.* (and following), *et al.* (and others), *passim* (throughout).

### 4.4 Algorithm implications

- **Fixed multi-word phrases** (e.g., "habeas corpus" must be matched as a unit, not as "habeas" + "corpus" independently).
- **Always lowercase** when not a citation first word.
- Stemming/lemma must not break these phrases: "amicus curiae" cannot be normalized to "amic curiae".
- Latin words in legal English often have **classical pronunciation** that diverges from English; phonetic matching is unreliable.
- Use **dictionary lookup** for canonical form; these phrases are finite (~200 in common use).

---

## 5. Astronomical Names

### 5.1 Standards

- **IAU** (International Astronomical Union) — sole authority for proper names and constellation boundaries (since 1930).
- **Working Group on Star Names (WGSN)** — catalog of IAU-approved star proper names (330+ as of 2018).
- Multiple naming systems coexist: proper names, Bayer designation, Flamsteed designation, variable star designation, catalog numbers (HD, HIP, GJ, HR, BD, etc.).

### 5.2 Bayer designation (Johann Bayer, 1603, *Uranometria*)

**Format:** `<Letter> <Constellation Genitive>`

- **Letter** is lowercase **Greek** (α, β, γ, δ, ε, ζ, η, θ, ι, κ, λ, μ, ν, ξ, ο, π, ρ, σ, τ, υ, φ, χ, ψ, ω) for bright stars, then **lowercase Latin** (b, c, d, e, f, g, h, i, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z) omitting j and v, then **uppercase Latin** A, B, C... (mostly for southern constellations by Lacaille, Gould).
- **Constellation** is in **GENITIVE case** (Latin possessive form), often abbreviated to **3 letters**.
- **Order** was traditionally by **apparent magnitude** (brightest first), but this is imperfect (30+ constellations have an Alpha that isn't the brightest).
- **Compound designations** for close stars: `π¹ Orionis`, `π² Orionis`, ... `π⁶ Orionis`.
- **Numeric superscripts** for disambiguation: `ρ¹ Cancri` is written `55 Cancri` (Flamsteed preferred when there's a Flamsteed number too).
- Borders reassigned by IAU 1930: some stars now in different constellations but retain old Bayer names.

### 5.3 Flamsteed designation (John Flamsteed, 1712)

**Format:** `<Number> <Constellation>`

- Numbers assigned (not by Flamsteed himself but by J. J. Lalande, 1783) by **increasing right ascension** within a constellation.
- Used when **no Bayer designation** exists, or when Bayer would have a numeric superscript.
- Examples: **61 Cygni** (famous for Bessel measuring its parallax, 1838), **51 Pegasi** (first Sun-like star with exoplanet, 1995), **55 Cancri** (Copernicus / α¹ Cancri), **70 Ophiuchi** (binary).

### 5.4 Variable star designation

- Format: `R <Constellation>` to `Z <Constellation>`, then `RR..ZZ` (no J), then `AA..QZ`, then `V335..V` (since 335 = 9+8+...+25 = beginning of triple-letter range).
- `R`, `S`, `T`, `U`, `V`, `W`, `X`, `Y`, `Z` are single letters.
- Examples: **R Cygni**, **RR Lyrae**, **VY Canis Majoris**, **T Tauri**.

### 5.5 Catalog designations (full-sky)

| Catalog | Format | Example |
|---------|--------|---------|
| Henry Draper (HD) | `HD <6-digit number>` | HD 209458 (Osiris' star, first exoplanet transit) |
| Hipparcos (HIP) | `HIP <6-digit number>` | HIP 11767 |
| Gliese (GJ) | `GJ <number>` / `Gliese <number>` | GJ 273 (Luyten's Star) |
| Bright Star (HR) | `HR <4-digit number>` | HR 8799 |
| Bonner Durchmusterung (BD) | `BD +<dec> <number>` | BD +49 399 |
| SAO | `SAO <number>` | — |
| 2MASS | `2MASS J<coords>` | — |
| Gaia DR3 | `Gaia DR3 <id>` | — |

### 5.6 10+ Real Examples (with structure)

| # | Common name | Bayer | Flamsteed | Other catalog | Type |
|---|-------------|-------|-----------|---------------|------|
| 1 | **Sirius** (α CMa) | α Canis Majoris | 9 Canis Majoris | HD 48915, HIP 32349 | Brightest star in night sky |
| 2 | **Betelgeuse** (α Ori) | α Orionis | 58 Orionis | HD 39801, HIP 27989 | Red supergiant |
| 3 | **Rigel** (β Ori) | β Orionis | 19 Orionis | HD 34085, HIP 24436 | Blue supergiant |
| 4 | **Polaris** | α Ursae Minoris | 1 Ursae Minoris | HD 8890, HIP 11767 | North Pole star |
| 5 | **Vega** | α Lyrae | 3 Lyrae | HD 172167, HIP 91262 | Once pole star (~12,000 BCE) |
| 6 | **Capella** | α Aurigae | 13 Aurigae | HD 34029, HIP 24608 | Yellow giant binary |
| 7 | **Aldebaran** | α Tauri | 87 Tauri | HD 29139, HIP 21421 | Orange giant |
| 8 | **Antares** | α Scorpii | 21 Scorpii | HD 148478, HIP 80763 | Red supergiant |
| 9 | **Spica** | α Virginis | 67 Virginis | HD 116658, HIP 65474 | Blue binary |
| 10 | **Altair** | α Aquilae | 53 Aquilae | HD 187642, HIP 97649 | A-type main sequence |
| 11 | **Deneb** | α Cygni | 50 Cygni | HD 197345, HIP 102098 | Blue-white supergiant |
| 12 | **Proxima Centauri** | (no Bayer) | (no Flamsteed) | GJ 551, HIP 70890 | Closest star to Sun |
| 13 | **61 Cygni** | (no Bayer) | 61 Cygni | HD 201091/201092, HIP 104214/104217 | First parallax measured (Bessel 1838) |
| 14 | **51 Pegasi** | (no Bayer) | 51 Pegasi | HD 217014, HIP 113357 | First exoplanet around Sun-like star (1995) |
| 15 | **55 Cancri** (= Copernicus) | ρ¹ Cancri | 55 Cancri | HD 75732, HIP 43587 | Star with 5 known exoplanets |
| 16 | **R Cygni** | (no Bayer) | (no Flamsteed) | HD 185456 | Variable star, single-letter designation |
| 17 | **VY Canis Majoris** | (no Bayer) | (no Flamsteed) | HD 38089, HIP 36846 | Variable, one of largest known stars |
| 18 | **Alpheratz** | α Andromedae | 21 Andromedae | HD 358, HIP 677 | Shared border star (Andromeda/Pegasus) |

### 5.7 Common name vs. scientific name

- **Common (proper) name**: "Polaris", "Sirius", "Betelgeuse" — colloquial, often Arabic-derived, IAU-standardized.
- **Bayer designation**: "Alpha Ursae Minoris" — Greek/Latin letter + Latin genitive.
- **Flamsteed**: "1 Ursae Minoris" — number + nominative.
- **Catalog**: "HD 8890" — ID only.

The same star has **multiple acceptable names**. Algorithms must know they're synonyms.

### 5.8 Edge cases

- **Constellation genitive** has unusual endings: *Centauri* (Centaurus), *Orionis* (Orion), *Ursae Minoris* (Ursa Minor), *Bootis* (Boötes), *Leonis* (Leo), *Aquilae* (Aquila).
- 3-letter **abbreviations** are not always the first 3 letters: *UMi* = Ursa Minor, *Ori* = Orion, *Aql* = Aquila, *Cnc* = Cancer.
- Roman-letter Bayer and Flamsteed-number: `55 Cancri` is preferred over `Rho-1 Cancri` (the "rho" requires a superscript).
- Lacaille used Latin letters three times over for **Argo Navis** (later split into Carina, Puppis, Vela).
- The **Greek letter omicron** (ο) and Latin letter **o** are visually identical; in some old atlases "o Scorpii" (Latin) was misinterpreted as omicron.

### 5.9 Algorithm implications

- Recognize **`<Greek|Latin letter> <3-letter genitive abbreviation>`** pattern.
- Recognize **`<number> <constellation nominative>`** for Flamsteed.
- Recognize **`<uppercase R..Z, RR..ZZ, V<n>> <constellation>`** for variable stars.
- Map common name ↔ Bayer ↔ Flamsteed ↔ catalog via a **synonym table**.
- Constellation genitive must be **resolvable from abbreviation**; build the table.

---

## 6. Diacritics in Names

### 6.1 Standards

- **Unicode** is the modern standard; **precomposed** (NFC) and **decomposed** (NFD) forms exist.
- **NFKC** (compatibility decomposition + canonical composition) is the recommended normalization for matching.
- **ISO 9**, **ALA-LC** (Library of Congress), and **DIN 31635** are transliteration standards for non-Latin → Latin scripts (e.g., Cyrillic → Latin).
- **MATHIR already has `normalize_unicode` (NFKC/NFD) and a `transliteration function`** per `SWARM_CONTEXT_LATIN.md`.

### 6.2 Common diacritics in European proper names

| Diacritic | Name | Example | Transliteration |
|-----------|------|---------|-----------------|
| `ä, ö, ü` | Umlaut / diaeresis | Müller, Schröder, Göttingen | ae, oe, ue |
| `é, è, ê, ë` | Acute, grave, circumflex, diaeresis (French) | François, Beyoncé, Château, Citroën | e (silent) |
| `á, í, ó, ú` | Acute (Spanish, Czech, Hungarian) | Sá, Yáñez, Gábor | a, i, o, u |
| `ñ` | Tilde (Spanish) | Ibañez, Mañana | n |
| `ç` | Cedilla (French, Portuguese) | Garçon, Praça | c |
| `ß` | Eszett (German) | Straße | ss |
| `ł` | Stroke / kreska (Polish) | Łukasiewicz, Łódź | l (in Latin script) or L-with-stroke (preserved) |
| `Ł` | Capital ł | Łódź | same |
| `ą, ę` | Ogonek (Polish) | Kraków region, Polish names | a, e (with nasal quality) |
| `ć, ń, ś, ź` | Acute (Polish) | Śląsk, Źrebce | c, n, s, z |
| `ó` | Acute (Polish) = /u/ sound | Kraków | o (in Polish, sounds like "u") |
| `ż` | Overdot (Polish) | Żuraw, Wałęsa | z |
| `ř, ť, ď, ň, č, š, ž, ě` | Caron (Czech, Slovak) | Dvořák, Čapek, Škoda, Smetana, Janáček | r, t, d, n, c, s, z, e |
| `đ` | Stroke (Croatian, Vietnamese) | Đorđe, Đà Nẵng | d (Croatian) or d (Vietnamese, with different sound) |
| `ø` | O-stroke (Danish, Norwegian, Faroese) | København, Søren | o |
| `å` | A-ring (Scandinavian) | Ångström, Oslo, Malmö | a (aa historically) |
| `æ` | Ash (Danish, Norwegian, Icelandic) | Æsop, Reykjavík, Schrödinger | ae |
| `þ, ð` | Thorn, eth (Icelandic) | Þórr, Eiður | th, d |
| `ı` | Dotless i (Turkish) | Isparta | i |
| `İ` | Dotted capital I (Turkish) | İstanbul, İzmir | I (capital) |
| `ğ` | Breve-Latin G (Turkish) | Eğitim, Ağaç | g (silent or lengthens vowel) |
| `ş` | S-cedilla (Turkish) | Şahin, Güneş | sh |
| `ç` | C-cedilla (Turkish) | Çelik, Çanakkale | ch |
| `ö, ü` | O/U-diaeresis (Turkish) | Öztürk, Gülçin | o, u (fronted) |
| `ŵ, ŷ` | Circumflex (Welsh) | Llanrwst, Ŵyn | w, y (long) |

### 6.3 10+ Examples with Transliteration Pairs

| # | Canonical | ASCII transliteration | Notes |
|---|-----------|----------------------|-------|
| 1 | **Erwin Schrödinger** | Erwin Schroedinger | ö → oe (also "Schrodinger") |
| 2 | **Wolfgang Amadeus Mozart** | Wolfgang Amadeus Mozart | ASCII safe |
| 3 | **Ludwig van Beethoven** | Ludwig van Beethoven | ASCII safe |
| 4 | **Antonín Dvořák** | Antonin Dvorak | š → sh, í → i, á → a |
| 5 | **Bedřich Smetana** | Bedrich Smetana | ř → r, č → ch |
| 6 | **Leoš Janáček** | Leos Janacek | š → sh, č → ch, á → a, ě → e |
| 7 | **Frédéric François Chopin** | Frederic Francois Chopin | é → e, ç → c |
| 8 | **Hermann Müller** | Hermann Mueller | ü → ue |
| 9 | **François Mitterrand** | Francois Mitterrand | ç → c |
| 10 | **José Carreras** | Jose Carreras | é → e |
| 11 | **Andrzej Wajda** | Andrzej Wajda | (already ASCII) |
| 12 | **Stanisław Lem** | Stanislaw Lem | ł → l, ław → law |
| 13 | **Wisława Szymborska** | Wislawa Szymborska | ł → l, ś → s, ą → a |
| 14 | **Lech Wałęsa** | Lech Walesa | ę → e, ł → l, ś → s (ą has no exact translit) |
| 15 | **Gösta Ekman** | Gosta Ekman | ö → o (in Swedish) or Gösta → Gosta (with diaeresis dropped) |
| 16 | **Søren Kierkegaard** | Soren Kierkegaard | ø → o, é → e |
| 17 | **København** | Kobenhavn / Copenhagen | ø → o |
| 18 | **Ångström** | Angstroem / Aangstroem | å → a (Swedish: treat as separate letter "Å" or "aa") |
| 19 | **René Descartes** | Rene Descartes | é → e |
| 20 | **Curaçao** | Curacao | ç → c |
| 21 | **François-Marie Arouet** (Voltaire) | Francois-Marie Arouet | ç → c |
| 22 | **Zürich** | Zuerich / Zurich | ü → ue, ü → u |
| 23 | **São Paulo** | Sao Paulo | ã → a (loses nasal quality) |
| 24 | **İstanbul** | Istanbul | İ → I (capital dot preserved) |
| 25 | **Łukasiewicz** | Lukasewicz (lossy) / Lukasiewicz | ł → l (lossy: Ł and L are distinct in Polish) |

### 6.4 When diacritics MATTER (false friends)

Some diacritics produce **distinct letters** in their native alphabet — collapsing them is a real correctness loss:

| Pair | Languages | Risk |
|------|-----------|------|
| **Ł** vs **L** | Polish | Two different letters. *Łódź* (city) ≠ *Lodz* (random letters). *Łukasz* (name) ≠ *Lukasz* (could be different person). |
| **İ** vs **I** | Turkish | Two different letters. *İstanbul* (city) ≠ *Istanbul* (different word in Turkish). |
| **ı** vs **i** | Turkish | *Isparta* (province) vs *ısparta* (could be a different word). |
| **Đ** vs **D** | Croatian, Vietnamese | *Đorđe* (name) ≠ *Dorde*. |
| **ø** vs **o** | Danish, Norwegian | Treated as **different letters** in alphabet. |
| **å** vs **a** | Swedish, Norwegian, Danish | Treated as **separate letter** in alphabet. |
| **ä, ö, ü** vs **a, o, u** | Estonian, Finnish, Swedish, German | In German, they're variants of a, o, u; in Estonian/Finnish/Swedish, they're **separate letters**. |
| **ñ** vs **n** | Spanish | *ñ* is the 15th letter of the Spanish alphabet. *año* (year) ≠ *ano* (anus). |
| **ř** vs **r** | Czech | *řeka* (river) ≠ *reka* (not a word) — losing the caron loses the sound. |
| **č, š, ž** vs **c, s, z** | Czech, Slovak, Slovenian, Croatian, Baltic | Different letters. |

### 6.5 Algorithm implications

- **Always normalize to NFC (or NFKC) Unicode form** before comparison. NFD loses the diacritic but preserves semantic.
- **Index both forms**: canonical-with-diacritics + ASCII-transliteration, so `Müller` matches `Mueller` AND `Muller` (typo of one letter).
- **Polish Ł/L is a special case**: do NOT collapse Ł → L; instead, store them as separate tokens. For matching, generate the Ł-removed AND L-removed variants.
- **Turkish I/i/İ/ı**: the four-way ambiguity is legendary. A simple `lower()` will map `İ → i̇` (I + combining dot), not `i`. Use `locale.str.lower` for Turkish or `I=ı, İ=i` (dotless rule).
- **German ß**: in modern orthography (post-2017), ß can be replaced by SS in capitals. Store both `Straße` ↔ `Strasse`.
- **Diacritic stripping is LOSSY**: never store the stripped form as the canonical form. Use it only as a search expansion.

---

## 7. Roman Numerals

### 7.1 Standard forms

- **I** = 1, **V** = 5, **X** = 10, **L** = 50, **C** = 100, **D** = 500, **M** = 1000.
- **Additive**: same-value letters add; larger-then-smaller letters add. `XXVII` = 10+10+5+1+1 = 27.
- **Subtractive**: smaller-before-larger subtracts. `IV` = 4, `IX` = 9, `XL` = 40, `XC` = 90, `CD` = 400, `CM` = 900.
- **Standard range**: 1 to 3,999. (`MMMCMXCIX` = 3,999)
- **Case-insensitive**: `mmxxvi` and `MMXXVI` are the same.

### 7.2 Common contexts

| Context | Example | Notes |
|---------|---------|-------|
| **Regnal / monarch names** | Henry VIII, Louis XIV, Charles IV, Pope Leo XIV | Read as ordinals; e.g., "the Eighth" |
| **Generational suffixes** | William Howard Taft IV, John Smith III | Read as ordinals |
| **Wars** | World War II, Franco-Prussian War (no Roman), Cold War (no Roman) | |
| **Super Bowls** | Super Bowl XLII (42), Super Bowl LVI (56) | NFL championship game |
| **Volumes / parts** | Volume III, Chapter IV, Part II | |
| **Sequels** | Star Wars Episode III, Rocky IV, Pope John Paul II | |
| **Centuries** (Romance) | French *xviiie siècle* (18th c.), Spanish *siglo XVIII* | Lowercase, often small caps |
| **Quadrants of a graph** | Quadrant I, II, III, IV (when axes can be negative) | |
| **Military units** | XVIII Airborne Corps, III Panzerkorps | |
| **Act / Scene** | Act III, Scene ii (lowercase Roman for scene in some conventions) | |
| **Outline levels** | I., II., III., A., B. (often with periods) | English legal/educational |
| **Sport teams** | 1st XV (rugby), 3rd XI (cricket) | Sometimes mixed with Arabic |
| **Outlines** | i., ii., iii. (lowercase for sub-sections) | |
| **Page numbers** (preface) | i, ii, iii, iv, v (front matter) | Often lowercase |
| **Blood types** | (No Roman — O, A, B, AB) | |
| **Repetition marks** | (No Roman — "ditto" or " " or ‡) | |
| **Year on monuments** | Anno Domini MMXXVI = AD 2026 | |
| **Copyright dates** | © 2024 (no Roman, modern practice) | |
| **Sport jerseys** | (No Roman — Arabic) | |

### 7.3 Variants & non-standard forms

| Variant | Standard equivalent | Notes |
|---------|---------------------|-------|
| **IIII** | IV = 4 | Common on **clock faces** (esp. Big Ben, Wells Cathedral clock). Also used by monarchs Louis XIV, Charles IV of Spain on coinage. Colosseum gates used IIII. |
| **VIIII** | IX = 9 | Used by Roman Senate on coins; medieval use |
| **XIIII / XIIII** | XIV = 14 | Medieval practice |
| **XIIX** | XVIII = 18 | Used by **XVIII Roman Legion**; reflects Latin *duodeviginti* (two from twenty). Also on Fasti Antiates calendar. |
| **IIXX** | XVIII = 18 | Used by **XXII Roman Legion** (twenty-second = *duo et vicesima* = "two and twentieth"). |
| **XCVXIX** | 1519 | French reading: *quinze-cent-dix-neuf* (fifteen-hundred-nineteen) |
| **MDCCCCX** | MCMX = 1910 | Used on Admiralty Arch, London (1910) |
| **MDCDIII** | MCMIII = 1903 | Saint Louis Art Museum entrance (rare) |
| **XIIII / XLIIII** | 44 | Colosseum gates 44 |
| **IIIXXXIX** | 99 | French *quatre-vingt-dix-neuf* (4×20+19) |
| **lower-case** | i, ii, iii, iv (1-4); but v, vi, vii, viii (5-8) | Lowercase often used in **outlines** and **preface pagination** |
| **Apostrophus** (medieval) | IↃ = 500, CIↃ = 1000, IↃↃ = 10,000, etc. | For large numbers, obsolete |
| **Vinculum** | Multiplies by 1000 | Historical: V̅ = 5000, X̅ = 10000 |

### 7.4 Edge cases & parsing rules

- A Roman numeral can be **substring** of a larger word: "Ivanka" contains "I", "V"; must be detected as context (separated by spaces, periods, or word boundaries).
- "Mi" or "MiX" or "Dix" can be misinterpreted (e.g., a French surname "Midi" = Roman "MIDI"?).
- A Roman numeral can be followed by a **lettercase change**: "Type III" (uppercase Roman) vs. "page iii" (lowercase Roman).
- A Roman numeral can be **in URL paths**: `/v1/v2/iii` — common in API versioning.
- A Roman numeral can be **in copyright notices**: "© MMXXIV" — but rarely in modern practice.

### 7.5 Algorithm implications

- **Tokenize Roman numerals as a separate token type** alongside words and numbers (Arabic).
- Provide a **canonical regex** for Roman numerals: `M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})` (canonical 1-3999).
- For string similarity, treat `IV` and `IIII` as **equivalent** by mapping to the integer (4).
- Lowercase / uppercase both work; canonicalize to uppercase for matching.
- Roman numerals in **names** (Henry VIII) should be treated as **part of the name** (do not normalize away).
- Roman numerals in **numeric position** (Chapter IV) can be normalized to Arabic for index sorting.

---

## 8. Universal Patterns for `latin_names.py`

Based on all 7 research areas, here are the **cross-cutting algorithmic patterns** an algorithm for MATHIR must handle:

### Pattern 1: Two- or three-token binomial (genus + species [+ subsp])
- Format: `[Capitalized Latin] [lowercase Latin] ( [lowercase Latin] )?`
- Tokens: 2-3, separated by spaces
- Authority/year may follow: `(Linnaeus, 1758)` or `Linnaeus, 1758`
- Examples: `Homo sapiens`, `Escherichia coli`, `Canis lupus familiaris`
- Algorithm: detect pattern, extract canonical form, drop authority

### Pattern 2: Stem-based pharmaceutical matching
- Format: arbitrary prefix + class-suffix
- Examples: `*-statin`, `*-pril`, `*-mab`, `*-olol`
- Algorithm: index INN stems as separate tokens; allow stem-based expansion (matches all drugs in class)

### Pattern 3: Multi-word fixed Latin phrases
- Format: 2-3 word unbreakable phrases
- Examples: `habeas corpus`, `pro bono`, `amicus curiae`, `mens rea`
- Algorithm: dictionary lookup; never stem or split

### Pattern 4: Letter + Latin genitive (Bayer) or Number + Nominative (Flamsteed)
- Format: `[Greek|Latin letter] [3-letter genitive]` OR `[number] [constellation]`
- Examples: `α CMa`, `Betelgeuse (α Ori)`, `61 Cygni`, `Polaris (α UMi)`
- Algorithm: parse letter/number, lookup constellation, expand to common name

### Pattern 5: Diacritic-aware matching with transliteration fallback
- Format: any word with combining diacritics
- Examples: `Müller`, `Schrödinger`, `Łukasiewicz`, `François`
- Algorithm: NFC/NFKC normalize; index both canonical and ASCII-transliteration; special-case Ł/ł (Polish) and I/İ/ı (Turkish)

### Pattern 6: Roman numeral tokenization
- Format: `M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})` at word boundary
- Examples: `VIII`, `XII`, `XLII`, `III` (lowercase)
- Algorithm: detect Roman numeral, canonicalize to uppercase, optionally expand to integer

### Pattern 7: Comma-separated author citation in taxonomy
- Format: `<name>, <4-digit year>` optionally parenthesized
- Examples: `(Linnaeus, 1758)`, `Linnaeus, 1758`
- Algorithm: split, retain or drop authority based on context

### Pattern 8: Latin noun-phrase anatomy
- Format: `[musculus|arteria|vena|os|...] [adjective] [optional adjective]`
- Examples: `musculus sternocleidomastoideus`, `arteria carotis communis`
- Algorithm: detect pattern, drop the structure-class noun to get the meat (`sternocleidomastoideus`), apply -us → -oid English shortening for matching

---

## 9. Edge Cases & Common Errors (Master List)

| # | Category | Issue | Fix |
|---|----------|-------|-----|
| 1 | Taxonomy | *E. coli* is shorthand for *Escherichia coli* | Expand genus-initial to full genus |
| 2 | Taxonomy | "(Linnaeus, 1758)" parentheses mean reclassification | Strip parens; they're metadata |
| 3 | Taxonomy | "subsp." vs "ssp." (botany vs. zoology) | Map both to canonical "subspecies" |
| 4 | Anatomy | "sternocleidomastoid" (English) vs "sternocleidomastoideus" (Latin) | -us → -oid shortening rule |
| 5 | Anatomy | "lat dorsi" (clinical slang) for *latissimus dorsi* | Dictionary lookup |
| 6 | Anatomy | Eponyms vs Latin terms (Broca's area vs *area Broca*) | Maintain eponym → Latin mapping |
| 7 | Pharma | "paracetamol" (INN/BAN) vs "acetaminophen" (USAN) | Stem-prefix match (`paracet-`, `acet-`) |
| 8 | Pharma | "amfetamine" (INN) vs "amphetamine" (BAN/USAN) | INN has `f`, regular has `ph` |
| 9 | Pharma | INN modified: "oxacillin sodium" (space-separated) | Treat as 2 tokens |
| 10 | Pharma | Brand name "Tylenol" ≠ INN "paracetamol" | Brand names should not be auto-mapped |
| 11 | Legal | "habeas corpus" split into "habeas" + "corpus" loses meaning | Fixed phrase dictionary |
| 12 | Legal | "amicus curiae" not the same as "amic curiae" | Never split |
| 13 | Legal | "infra", "supra", "ibid." are cross-references | Preserve as-is |
| 14 | Astronomy | "Polaris" vs "Alpha Ursae Minoris" vs "HR 489" vs "HD 8890" | Synonym table |
| 15 | Astronomy | Constellation genitive differs from nominative | Genitive table (e.g., *Orion* → *Orionis*) |
| 16 | Astronomy | "Rho-1" vs "55" Cancri (Flamsteed preferred) | Use most-common designation |
| 17 | Diacritics | "Müller" / "Mueller" / "Muller" are same name | Index all three |
| 18 | Diacritics | Polish "Łukasz" ≠ "Lukasz" (Ł is its own letter) | Don't collapse Ł to L |
| 19 | Diacritics | Turkish "İstanbul" ≠ "Istanbul" (İ is dotted) | Locale-aware lower() |
| 20 | Diacritics | "ß" / "ss" in German | Index both forms |
| 21 | Diacritics | Spanish "año" (year) ≠ "ano" (anus) | Don't strip ñ |
| 22 | Roman numerals | "Henry VIII" should not be "Henry 8" (it's a name) | Keep Roman as part of name |
| 23 | Roman numerals | "IV" vs "IIII" (clock face variant) | Both map to 4 |
| 24 | Roman numerals | "I" in URL path = version 1, not Roman | Context-dependent |
| 25 | Roman numerals | "XIIX" (Latin duodeviginti) maps to 18 | Use extended parser |
| 26 | Roman numerals | Lowercase "iii" = 3, "iv" = 4 | Case-insensitive |
| 27 | All | Diacritics in URLs: `/müller` → `/mueller` after IDN encoding | Punycode (xn--mller) |
| 28 | All | Case insensitivity for ALL letter categories | unicode-aware lower-casing |
| 29 | All | Unicode normalization (NFC, NFD, NFKD, NFKC) | Always NFC before comparison |
| 30 | All | Cross-script: "Müller" (Latin) vs "Мюллер" (Cyrillic) | Transliterate to common script first |

---

## 10. Sources

### Primary (Tier 1)

- **International Code of Zoological Nomenclature (ICZN)** — 4th edition, 1999, online at https://www.iczn.org/
- **International Code of Nomenclature for algae, fungi, and plants (ICN)** — Shenzhen Code, 2018, https://www.iapt-taxon.org/nomen/main.php
- **Terminologia Anatomica** — 2nd ed., 2019, Thieme, https://www.thieme.com/books-main/clinical-neurology/product/4373-terminologia-anatomica
- **WHO INN Programme** — https://www.who.int/teams/health-product-and-policy-standards/inn
- **WHO Stem Book** — "The use of stems in the selection of International Nonproprietary Names (INN) for pharmaceutical substances" (2024) — https://iris.who.int/bitstream/handle/10665/379226/9789240099388-eng.pdf
- **International Astronomical Union (IAU)** — https://www.iau.org/
- **Unicode Standard** — chapters 3, 4, 5, 7 (Normalization, Case Folding, Diacritics)
- **ISO 9:1995** — Transliteration of Cyrillic
- **ALA-LC Romanization Tables** — Library of Congress

### Secondary (Tier 2) — used in this report

- Wikipedia: *Binomial nomenclature* — https://en.wikipedia.org/wiki/Binomial_nomenclature
- Wikipedia: *Author citation (zoology)* — https://en.wikipedia.org/wiki/Author_citation_(zoology)
- Wikipedia: *International nonproprietary name* — https://en.wikipedia.org/wiki/International_Nonproprietary_Name
- Wikipedia: *Terminologia Anatomica* — https://en.wikipedia.org/wiki/Terminologia_Anatomica
- Wikipedia: *Anatomical terminology* — https://en.wikipedia.org/wiki/Anatomical_terminology
- Wikipedia: *Bayer designation* — https://en.wikipedia.org/wiki/Bayer_designation
- Wikipedia: *Flamsteed designation* — https://en.wikipedia.org/wiki/Flamsteed_designation
- Wikipedia: *Stellar designations and names* — https://en.wikipedia.org/wiki/Stellar_designations_and_names
- Wikipedia: *List of legal Latin terms* — https://en.wikipedia.org/wiki/List_of_legal_Latin_terms
- Wikipedia: *Diacritic* — https://en.wikipedia.org/wiki/Diacritic
- Wikipedia: *Polish alphabet* — https://en.wikipedia.org/wiki/Polish_alphabet
- Wikipedia: *Roman numerals* — https://en.wikipedia.org/wiki/Roman_numerals

### Tertiary (Tier 3) — referenced but not authoritative

- Stack Overflow threads on Unicode normalization (NFKC vs NFC)
- GitHub issues on `unidecode` library (lossy transliteration)
- GBIF backbone taxonomy (Global Biodiversity Information Facility)
- ITIS (Integrated Taxonomic Information System)
- WGSN Bulletin (IAU Working Group on Star Names)

---

## 11. Confidence Level

**High confidence** on:
- Binomial nomenclature structure (very well documented; ICZN rules are explicit)
- INN stem system (WHO official)
- Bayer/Flamsteed designation structure (extensive documentation)
- Roman numerals (well-established standard)
- Anatomical terms word formation (TA standard)
- Legal Latin (curated list available)

**Medium confidence** on:
- Specific diacritic handling edge cases (Polish Ł/L, Turkish I/i — many gotchas)
- Some anatomical eponyms (regional variation)
- Some compound Bayer/Flamsteed edge cases (rare historical forms)

**Lower confidence** on:
- Real-world frequency of historical Roman-numeral variants (XIIX, IIXX) in modern text — likely very low
- Specific INN stem lists beyond the common 20-30 (full list is 400+ in *Stem Book*)
- Modern usage of obscure legal Latin phrases (most are post-2010 only used in academic writing)

---

## 12. Recommended Next Steps

1. **Build a Latin name dataset**: 200+ canonical examples covering all 7 topics (curate from Wikipedia, WHO INN list, GBIF, IAU WGSN).
2. **Implement the 8 patterns** from §8 in `latin_names.py` with explicit unit tests.
3. **Build synonym tables**:
   - Eponyms → Latin TA term (e.g., "Broca's area" → "area Broca" / "gyrus frontalis inferior pars opercularis")
   - INN ↔ USAN ↔ BAN (paracetamol ↔ acetaminophen)
   - Bayer ↔ Flamsteed ↔ proper name (α UMi = 1 UMi = Polaris)
   - Diacritic canonical ↔ ASCII transliteration
4. **Add Roman-numeral detection** to the tokenizer so "Henry VIII" doesn't get split into "Henry" and "VIII" (or alternatively, treat VIII as an integer 8 but preserve the surface form for display).
5. **Use NFKC normalization** as the default preprocessing step.
6. **Add Polish Ł/L and Turkish I/İ/ı special handling** — these are the most-mis-handled cases.
7. **Integration test against UNIBRI** to verify that "Müller" and "Mueller" now match at the same recall level.
