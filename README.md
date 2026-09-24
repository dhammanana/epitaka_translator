# Epitaka Translator

> Databases (`epitaka.db`, `dpd-dictionary.db`, `epitaka_en/th/si.db`,
> `epitaka_my_nissaya.db`):
> https://github.com/dhammanana/epitaka_app/releases/tag/latest

Translate Pāli Theravāda books (Tipiṭaka, commentaries, sub-commentaries) into
modern languages with Gemini, one book at a time. Translations accumulate in
per-language SQLite files that the Epitaka app / web server reads directly.

## How it works

`src/book_translator.py` translates a whole book, part by part:

1. **Find pending lines.** For each `book_id`, every `(para_id, line_id)` not
   yet present in `epitaka_<lang>.db` is pending (unless `--overwrite`).
2. **Section by headings.** Paragraphs are grouped into sections along heading
   boundaries (`min_lines` merged), then small sections are merged again so
   one API call covers up to ~50 pending lines.
3. **Chunk token-safely.** Each section is split into chunks of at most
   `--max-tokens` (default 3000) of Pāli text; tiny leftover chunks are
   folded into a neighbour instead of firing their own call.
4. **Build a rich prompt** (`src/common/context_builders.py`) with up to
   8 blocks: established glossary, Pāli commentary/ṭīkā, word definitions,
   previous paragraph, translated mūla/aṭṭhakathā references, parallel human
   translations (English/Thai/Sinhala), Myanmar nissaya gloss, and the
   sentences to translate.
5. **Call Gemini.** The model must return JSON with exactly three keys:
   `translations` (one per sentence, each with `confidence: high|low` and an
   optional `confidence_note`), `glossary` (new Pāli-stem → translation
   terms), `remarks` (genuine conflicts between sources only).
6. **Script-bleed guard.** Because the prompt contains Thai/Sinhala/Myanmar
   reference text, every returned translation/glossary term is scanned for
   Thai, Sinhala, Myanmar, and Devanagari script that does not belong to the
   target language. Contaminated items are **dropped, not saved** — the lines
   stay pending and are retried on a later run.
7. **Save.** Translations (+ confidence) go to `epitaka_<lang>.db/sentences`,
   remarks to `epitaka_<lang>.db/translation_remarks`, new terms to
   `glossary_<lang>.db/glossary` (Pāli stems resolved against the dictionary
   tables, grammar particles skipped, duplicates upserted).

Re-running the same command only translates what is still missing, so runs
are resumable and `--books preset` can loop for days via `runner.sh`.

## Translation context: how the data helps

A bare Pāli sentence is often ambiguous — compounds can split several ways,
technical terms have doctrine-specific senses, and commentaries resolve what
the bare text leaves open. So every chunk is sent to the model together with
reference context assembled by `src/common/context_builders.py`. Each block
has one job:

| Prompt block (builder) | Source data | Job |
|---|---|---|
| `PALI COMMENTARY & SUB-COMMENTARY` (`CommentaryContext`) | `book_links` + `sentences` in `epitaka.db` | Nail the **meaning** |
| `TRANSLATED MŪLA / AṬṬHAKATHĀ / ṬĪKĀ REFERENCES` (`MulaAtthaContext`) | linked paras + `epitaka_<lang>.db`, fallback `epitaka_en.db` | Nail the **context and terms** when translating commentaries |
| `ESTABLISHED GLOSSARY` (`GlossaryContextStemmed`) | `glossary_<lang>.db` + dictionary tables | Nail **term consistency** |
| `PARALLEL HUMAN TRANSLATIONS` (`ParallelTranslationContext`) | `epitaka_en/th/si.db` | Second opinions + conflict flagging |
| `PALI WORD DEFINITIONS` (`PaliDefsContext`) | `pali_definition` + `frequency_word.db` | Explain **difficult words** |
| `MYANMAR NISSAYA` (`NissayaContext`) | `epitaka_my_nissaya.db` | Word-by-word gloss |
| `PREVIOUS PARAGRAPH` (`PreviousTranslationContext`) | `epitaka_<lang>.db`, fallback `epitaka_en.db` | Tone/terminology continuity |

### Commentary nails the meaning

`CommentaryContext` walks the `book_links` table (which maps each line to
the lines that explain it) in three directions and renders every line as
Pāli plus its available translation:

- **Forward** — translating mūla/aṭṭhakathā: pulls the commentary and
  sub-commentary paragraphs that annotate these exact lines. Three size
  tiers keep it within budget: full paragraphs → 3-line window around each
  linked line → linked lines only.
- **Reverse** — translating a commentary/ṭīkā: pulls back the **mūla source
  paragraphs** it comments on, so the translator sees what is being explained.
- **Sibling** — pulls the *other* commentaries on the same mūla (e.g. when
  translating a ṭīkā, also shows the aṭṭhakathā on the same root lines),
  giving the full commentary stack.

The system prompt declares this block the **primary authority** for hard
compounds, technical terms, and ambiguous syntax.

### Mūla + existing translation nails commentary wording

When the chunk being translated *is* a commentary, `MulaAtthaContext` adds
the linked mūla paragraphs as Pāli **plus their existing translation in the
current target language** (`epitaka_<lang>.db`), falling back to the English
reference (`epitaka_en.db`) where the target language has nothing yet. If the
commentary defines or quotes a root-text term, the prompt instructs the model
to reuse the established target-language rendering of that term — keeping
each commentary consistent with the root passage it explains.

### Glossary nails consistency (known weak spot)

`GlossaryContextStemmed` looks up the chunk's words in `glossary_<lang>.db`
(the shared translation memory) and tells the model to apply them exactly:

- Each word is resolved to its **dictionary stem** via three tables in order
  — `dpd_inflections_to_headwords` → `pali_definition` → `dpr_stem` — so an
  inflected form like *buddhassa* still matches the stem entry *buddha*.
- Multi-word phrases are matched as raw **n-grams up to 5 words** (n = 2–5),
  catching set phrases such as *namo tassa bhagavato arahato
  sammāsambuddhassa*.
- To fight bloat, at most **3 variants per term** are shown; further ones are
  replaced with a note telling the model to reuse one instead of minting
  another. Grammar particles (*ca, pi, eva*, …) are never stored.

Honest status: this block underperforms today — the model still sometimes
renders a term with a new wording instead of the glossary one. Part of the
cause was mechanical and is now fixed: the prompt builder briefly wasn't
receiving the glossary path at all, so prompts went out with an empty
glossary block (visible in the older `examples/`). Remaining mitigations are
the 3-variant cap, the "reuse rather than add" prompt rules on both
the translation and glossary sides, and duplicate upserts on save. Improving
glossary adherence is the main open item in prompt quality.

### Parallel translations: English, Thai, Sinhala

`ParallelTranslationContext` fetches the **same exact lines** from the three
human-translation databases — English (also machine-assisted with a stronger
model plus SuttaCentral sources), Thai (Mahāmakuṭa edition), and Sinhala
(Tipiṭaka.lk) — strictly for meaning and terminology reference (the prompt
warns the model never to leak their wording or script into its output). The
model is additionally instructed to compare them: genuine disagreements
between the sources go into the `remarks` output with a suggested choice and
reason, saved to `translation_remarks` for **human review**.

### Word definitions for difficult words

`PaliDefsContext` handles the hard vocabulary:

1. Candidate words (≥ 5 letters) are ranked by **rarity across the whole
   Tipiṭaka** using the cached `frequency_word.db` (~963k words, built once),
   so rare, genuinely difficult words win over common grammar.
2. The **50 rarest** are kept; each is resolved to up to **3 dictionary
   headwords** (`pali_definition`, prefix search on the stem).
3. Each headword is shown with real Tipiṭaka usage sentences plus their
   translation — i.e. what the word means *in context*, not just a gloss.

These are typically the difficult/defined terms (often the bolded headwords
in commentarial definitions), which is why the translation prompt has a
special rule for rendering `<b>`-wrapped Pāli terms.

### Nissaya + previous paragraph

- `NissayaContext` adds the Myanmar **word-by-word gloss** (romanised to
  IAST) for each line, with edition labels where several exist — the most
  literal layer of help, and a listed trigger for `low` confidence when it is
  missing.
- `PreviousTranslationContext` carries the immediately preceding paragraph's
  translation forward for tone and terminology continuity.

### Example prompts & responses

The `examples/` folder holds three genuine prompt/response pairs from a
Vietnamese run (no trimming — exactly what was sent and received):

| Prompt | Response | Chunk | Sent / glossary / remarks |
|---|---|---|---|
| `20260924_064651_M-ii_p76-91_L1-7_prompt.txt` (127 KB) | `..._response.txt` (14 KB) | M-ii §§76–91, 7 lines | 49 translations / 5 glossary / 0 remarks |
| `20260924_064657_Ja-a-i_p1056-1079_L1-11_prompt.txt` (137 KB) | `..._response.txt` (27 KB) | Ja-a-i §§1056–1079, 11 lines | 61 translations / 6 glossary / 0 remarks |
| `20260924_064702_M-ii_p92-106_L1-17_prompt.txt` (115 KB) | `..._response.txt` (11 KB) | M-ii §§92–106, 17 lines | 49 translations / 0 glossary / 0 remarks |

File naming is `<timestamp>_<book>_p<paras>_L<lines>_{prompt,response}.txt`.
Read a prompt top-down to see every context block from the table above in
its real, full-size form; read the matching response for the
`translations` / `glossary` / `remarks` JSON the model must return (note:
real model output is occasionally loose JSON, e.g. unescaped quotes —
`parse_ai_json_response` in `ai_client.py` salvages complete items instead
of discarding the chunk).

One caveat: these examples were captured before a bug fix, so their
`ESTABLISHED GLOSSARY` block reads `(no glossary DB configured)` — the
prompt builder wasn't receiving the glossary path (see "Glossary" above).
Current code fills that block; expect it populated in newly generated logs.

Generate your own pair for any chunk with `--log-dir`:

```bash
python src/book_translator.py --lang si --books Dhp --max-parts 1 --log-dir ./examples
```

Related tools in `src/`:

| Script | Purpose |
|---|---|
| `book_translator.py` | Main translation run (this README's focus) |
| `glossary_builder.py` | Extract glossary terms from *already-translated* books (no new translations); `--check-only` flags suspect translations |
| `verify_translation.py` | Spot-check translation quality via a second model |
| `cleanup_bleeding.py` | Remove wrong-script rows already stored in a DB |
| `study_builder.py` | Generate English study guides into `epitaka_en.db` |

Shared infrastructure lives in `src/common/`: `common_utils.py` (DB paths,
schema, glossary upserts, stem lookup, script-bleed detection),
`ai_client.py` (all Gemini calls, key rotation, retries, JSON parsing),
`context_builders.py` (prompt context blocks).

## Setup

```bash
cd translator

# 1. Environment + dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. API keys
cp .env.example .env
# then edit .env and add at least one GEMINI_KEY_<N> (see below)

# 3. Data files (optional — runner.sh downloads these automatically)
#    ./data/epitaka.db              source Pāli texts + dictionary tables
#    ./data/dpd-dictionary.db       Pāli dictionary (stem lookup)
#    ./data/epitaka_en.db           English human translations (reference)
#    ./data/epitaka_th.db           Thai human translations (reference)
#    ./data/epitaka_si.db           Sinhala human translations (reference)
#    ./data/epitaka_my_nissaya.db   Myanmar nissaya word-by-word gloss
```

### `.env` and API key rotation

Get keys from [Google AI Studio](https://aistudio.google.com/apikey)
(one per Google account / project) and store them as numbered variables:

```dotenv
GEMINI_KEY_1="AIzaSy..."
GEMINI_KEY_2="AIzaSy..."
GEMINI_KEY_3="AQ.Ab8..."
```

How rotation works (`src/common/ai_client.py`, class `KeyRotator`):

- Keys are used **round-robin**. Numbering may have gaps; sorting is numeric
  (`KEY_2` before `KEY_10`). Labels in logs/state are the env var names, so
  you can tell exactly which key was parked or removed.
- Each key gets a budget of `AI_RPM_LIMIT` requests and `AI_TPM_LIMIT`
  tokens per rolling 60 s window (defaults `3` / `250000`, via env vars).
  When all keys are temporarily over budget the worker **waits** for the
  earliest one to free up — this is normal, not an error.
- Usage is tracked in a per-model JSON state file in the OS temp dir
  (`gemini_key_rate_state_<model>.json`, override with `AI_KEY_STATE_FILE`),
  guarded by a file lock, so **several processes sharing the same keys
  throttle each other correctly**.
- A key that returns HTTP 429 three times **for the same model** is parked
  for that model only (it can still serve other models). Keys rejected as
  invalid (401/403-type) are removed everywhere and recorded in the state
  file with label, reason, and timestamp.
- Models form a fallback pool (`FALLBACK_MODEL_CHAIN`, first = preferred).
  When every key is parked for the current model, it is dropped and the next
  model takes over. When nothing is usable at all, the run **stops loudly**
  ("All API keys exhausted", optional Telegram alert via `TELEGRAM_TOKEN` /
  `TELEGRAM_CHAT_ID`) — `runner.sh` then sleeps 3 h and retries.

## Running

All commands run from this `translator/` directory (scripts live under
`src/` since the reorganisation):

```bash
source .venv/bin/activate

# Translate into Sinhala, full preset book order, pinned model:
python src/book_translator.py --lang si --books preset --model gemini-3.7-flash

# Same, but with the automatic model fallback chain (recommended):
python src/book_translator.py --lang si --books preset

# Just two books, paragraphs 1-700, 4 paragraphs per chunk:
python src/book_translator.py --lang en --books Sp-i,Sp-ii --start 1 --end 700

# Re-translate everything (default resumes only missing lines):
python src/book_translator.py --lang th --books preset --overwrite

# Hands-off mode: data check + 3h-retry loop on key exhaustion:
./runner.sh si gemini-3.7-flash
./runner.sh si            # fallback model chain
```

Useful flags: `--start/--end` (para range, `-1` = end of book),
`--min-lines` (section size, default 50), `--max-tokens` (chunk budget,
default 3000), `--max-parts` (stop after N sections — for testing),
`--dry-run` (print prompts, call nothing), `--epitaka-db` / `--glossary-db`
(path overrides), `--log-dir` (prompt/response audit logs),
`--api-keys` (comma-separated keys instead of env vars).

### Languages (`--lang`)

Any code in `LANG_NAMES` (`src/common/common_utils.py`) works; the name is
injected into the prompt so the model knows its target language. English,
Thai, and Sinhala additionally serve as **parallel-translation references**
for every other language.

| Code | Language | Code | Language | Code | Language |
|---|---|---|---|---|---|
| `en` | English | `th` | Thai (ภาษาไทย) | `si` | Sinhala (සිංහල) |
| `ta` | Tamil | `hi` | Hindi | `ne` | Nepali |
| `bn` | Bengali | `mr` | Marathi | `gu` | Gujarati |
| `pa` | Punjabi | `te` | Telugu | `kn` | Kannada |
| `ml` | Malayalam | `or` | Odia | `lo` | Lao |
| `km` | Khmer | `my` | Burmese | `vi` | Vietnamese |
| `id` | Indonesian | `ms` | Malay | `tl` | Filipino (Tagalog) |
| `zh` | Chinese Simplified | `ja` | Japanese | `ko` | Korean |
| `de` | German | `fr` | French | `es` | Spanish |
| `pt` | Portuguese | `it` | Italian | `nl` | Dutch |
| `pl` | Polish | `ru` | Russian | `uk` | Ukrainian |
| `tr` | Turkish | `el` | Greek | `ro` | Romanian |
| `cs` | Czech | `hu` | Hungarian | `sv` | Swedish |
| `da` | Danish | `fi` | Finnish | `no` | Norwegian |
| `ar` | Arabic | `he` | Hebrew | `fa` | Persian |

### `--books preset` order

`preset` is the full `PRESET_BOOKS` list in `src/book_translator.py`,
ordered **mūla → aṭṭhakathā → ṭīkā → ancillary texts**: the Tipiṭaka
(Vinaya, Sutta, Abhidhamma) first, then commentaries, then
sub-commentaries, then histories, grammars, and handbooks. Order matters:
later books reuse earlier books' translations as in-prompt reference
(translated mūla/aṭṭhakathā context, glossary memory), so translating out of
order gives poorer terminology consistency.

## Data files and downloads

`runner.sh` checks `./data/` (override with `DATA_DIR=...`) for these
files at startup and downloads + unzips anything missing from the
[epitaka_app releases](https://github.com/dhammanana/epitaka_app/releases):

| Required file | Downloaded zip |
|---|---|
| `epitaka.db` | `epitaka.zip` |
| `dpd-dictionary.db` | `dpd-dictionary.zip` |
| `epitaka_en.db` | `epitaka_en.zip` |
| `epitaka_th.db` | `epitaka_th.zip` |
| `epitaka_si.db` | `epitaka_si.zip` |
| `epitaka_my_nissaya.db` | `epitaka_my_nissaya.zip` |

It needs `curl` (or `wget`) and `unzip`. `EPITAKA_DB` env var (or
`--epitaka-db`) overrides the source DB path; `epitaka_<lang>.db` and
`glossary_<lang>.db` are derived from it automatically. If `./data/` has no
`epitaka.db` yet but the legacy `../data/` (epitaka app folder) does, that
legacy folder is used as a fallback so existing checkouts keep working.

## Layout

```text
translator/
├── runner.sh            # data check + continuous translation loop
├── requirements.txt     # pip dependencies
├── .env.example         # copy to .env, add GEMINI_KEY_<N>
├── README.md            # this file
├── data/                # SQLite DBs (downloaded by runner.sh if missing)
├── examples/            # real prompt/response pairs (see above)
└── src/
    ├── book_translator.py    # main translation run
    ├── glossary_builder.py   # glossary from existing translations
    ├── study_builder.py      # English study guides
    ├── verify_translation.py # quality spot-checks
    ├── cleanup_bleeding.py   # wrong-script cleanup
    └── common/
        ├── common_utils.py     # DB paths/schema, glossary, script-bleed
        ├── ai_client.py        # Gemini calls, key rotation, retries
        ├── ai_client_bai.py    # alternate provider client (verification)
        └── context_builders.py # prompt context blocks
```
