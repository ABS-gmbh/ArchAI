# Entity extraction benchmark

`gold.json` marks the named entities - people, places, works, dates - in the
five reference transcriptions of `eval/quality_gate/readings.json`.
`scripts/benchmark_entities.py` runs each page through the full-page
pipeline's mention extractor and scores every mention of those types: a
mention is found when it overlaps a gold span of the same type, and anything
else is a false positive. Titles and roles are not scored.

```bash
python scripts/benchmark_entities.py
```

| Page | Entities |
|---|---|
| `old-french` | none |
| `latin-abaton` | none |
| `latin-apocalypse` | *iohannis*; *hyerusalem* |
| `middle-french-antitus` | *La satyre megere*; *Ame de monfalcon*; *Antitus*; *lausanne*; *lan de grace Mille cinq cens* |
| `old-french-verse` | *Caldꝰ*; *Marcadigas*, four times; *Ynabele*; *despaigne* (d'Espaigne) |

That is 14 entities. On the two pages with none, every mention is wrong.

## Caveats

- **The annotation is not expert.** It was made for this benchmark and has
  not been checked by a specialist. *nr̃e seigneur* (God) is not annotated.
- **One page was tuned on.** The extractor's anchors were written for
  `middle-french-antitus` (Lausanne, *pere*, *prince*, *euesque*), so the
  script also reports totals without it.
- **The pages were also the development set.** The extraction rules were
  developed while looking at these pages. Precision was therefore also
  checked on 72 transcriptions held out from a pipeline database, by reading
  every mention extracted from them (see the PR that added this benchmark).
- **Five pages, 14 entities.** That is enough to see the extractor fail on
  diplomatic text and to check the fix. It is not a corpus-level score.

## Linking benchmark

`scripts/benchmark_linking.py` scores authority linking on the same pages.
Each gold entity's `authority` in `gold.json` lists the Wikidata QIDs and
VIAF ids a correct link may name:
- `status: none` marks an entity with no authority record, where the right
  answer is no link;
- `status: n/a` marks the date, which is not scored.

Each page with entities is loaded into a throwaway pipeline database and
authority cache in a temporary directory. Mentions are extracted as the
pipeline extracts them, and `run_authority_linking` runs on them. Each gold
entity is then scored:
- `correct`: linked to an accepted id, or left unlinked when it has none;
- `wrong`: linked to anything else;
- `abstained`: a mention was found but not linked;
- `missed-mention`: no extracted mention overlaps it.

```bash
python scripts/benchmark_linking.py            # replay eval/entities/linking_cache.json, offline
python scripts/benchmark_linking.py --record   # ask Wikidata and VIAF again and rewrite the cache
```

`linking_cache.json` (about 320 kB) holds the 45 responses recorded on
2026-11-23 with `--record`, which uses the app's User-Agent, sends at most one
request a second, and stops on a 429 or 403. They are trimmed to what the
linker reads:
- of a Wikidata entity: labels, descriptions, aliases, and the main values
  of P31, P214, P1566, P625, P17 and P131;
- of a VIAF cluster: viafID, nameType, and the text of its headings and
  titles.
Replaying the trimmed cache gives the same result per entity as the live run.
A request missing from the cache is answered as a failed one and reported.

### How the gold ids were chosen (looked up 2026-11-23)

Each id comes from a live Wikidata `wbsearchentities` or `wbgetentities`
request, or a VIAF AutoSuggest request. VIAF ids are taken from the Wikidata
item's P214 where it has one.

| Entity | Accepted | Notes |
|---|---|---|
| *iohannis* | Q328804 John the Evangelist, Q44015 John the Apostle; VIAF 40648922, 101370748, 33149542580900300446 | The text says "apostle and evangelist John", so either item is right. |
| *hyerusalem* | Q1218 Jerusalem, Q567888 New Jerusalem; VIAF 131280745 | **Unsure.** Rev 21:10 describes the holy city coming down from heaven, so New Jerusalem is defensible too. |
| *La satyre megere* | Q135111996 | Item for the work by Antitus Faure (its P50). VIAF AutoSuggest found nothing. |
| *Ame de monfalcon* | Q678290; VIAF 58154739800452991157 | Aymon de Montfalcon, 1443-1517, bishop of Lausanne. |
| *lausanne* | Q807; VIAF 147349797 | The city. |
| *Antitus* | Q30729001 Antitus Faure; VIAF 234901669, 68994987 | **Unsure about 68994987**, a second VIAF cluster ("Antitus, -approximately 1501") that is probably the same person. |
| *lan de grace Mille cinq cens* | n/a | A date; not scored. |
| *Caldꝰ* | none | A romance king. Wikidata "Caldus" finds only a Roman cognomen and a Roman soldier, which would be wrong. |
| *Marcadigas* (4) and *Ynabele* | none | Romance characters. No Wikidata (en, fr) or VIAF result. |
| *despaigne* | Q29 Spain, Q12837 Iberian Peninsula; VIAF 133609710, 238991012 | Medieval *Espaigne* is closer to the peninsula than to the modern state. Both are accepted. |

### Result

Replayed from the cache; 13 scored entities:

| Page | correct | wrong | abstained | missed-mention |
|---|---|---|---|---|
| latin-apocalypse (n=2) | 0 | 0 | 1 (*iohannis*) | 1 (*hyerusalem*) |
| middle-french-antitus (n=4) | 2 (*lausanne* -> Q807, *La satyre megere* -> Q135111996) | 0 | 0 | 2 (*Ame de monfalcon*, *Antitus*) |
| old-french-verse (n=7) | 5 (the romance names, correctly unlinked) | 0 | 0 | 2 (*Ynabele*, *despaigne*) |
| total (n=13) | 7 | 0 | 1 | 5 |

The linker never linked a wrong record. Of the 7 entities that have a
record, it linked 2. The other 5 were missed at extraction (4) or abstained
on (1).

History:
- When the benchmark was added, the result was 6 / 0 / 2 / 5. *La satyre
  megere* abstained at 0.674: the Wikidata item has no aliases, and a missing
  alias scored as a 0.0 match.
- An alias-less candidate is now scored on its label: 6 -> 7 correct,
  2 -> 1 abstained, wrong 0 -> 0.
- The VIAF fix (fetching clusters from the documented endpoint) changed
  nothing here. The code before it gives the same outcomes; its 5 record
  fetches went to the retired viaf.json URL.

Thirteen entities on three pages is not a corpus-level score.

### Tried, not shipped

- **Trigger-free query variant** (2026-11-23). For a mention extracted by a
  trigger rule, search the bare name first, dropping a leading trigger word
  such as *euangelistae*, *sanctus*, *beati* or *roys*. On the five pages, 3
  mentions come from trigger rules, and the variant changes the queries of 2.
  It added 10 candidates, all for *euangelistæ iohannis* via the query
  "iohannis", and all wrong: Jan Hus, Leo Africanus, John Scotus Eriugena,
  Iohannes Caesariensis, Homann, Johann Müller, Linschoten, a Latin epic,
  a surname item, a Swiss village. The best scored 0.672, 0.13 under the
  0.80 threshold. No link changed and the right John never appeared:
  Wikidata's search for the Latin genitive does not reach Q328804 or Q44015.
  Risk without gain. What *iohannis* needs is Latin case normalisation
  (genitive to nominative, *Iohannes* / *John*), which waits for a held-out
  split.
- **Rules for the other misses** (*hyerusalem*, *Antitus*, *Ynabele*: no
  rule proposes them; *Ame de monfalcon*: a name split across lines;
  *despaigne*: d' written fused). Each would be written for these pages. They
  wait for held-out pages with ground truth, which do not exist under eval/
  yet.

## End-to-end benchmark

`scripts/benchmark_end_to_end.py` scores the whole path, from page image to
linked names. For each reference page it runs the full-page route's own
function (`ocr_router._extract_full_page`: the segmented engine, then
`_run_full_page_post_ocr_pipeline`, with the quality gate, mention
extraction, authority linking and indexing). It then runs the same post-OCR
pipeline on the reference transcription, so the table separates what OCR
errors cost from what extraction and linking cost. Every run gets a throwaway
database, vector store, authority cache and task directory under `--out-dir`.

```bash
python scripts/benchmark_end_to_end.py --images DIR --out-dir OUT --record   # live Wikidata/VIAF, saved to OUT/authority_responses.json
python scripts/benchmark_end_to_end.py --images DIR --out-dir OUT --cache OUT/authority_responses.json
python scripts/benchmark_end_to_end.py --reference-only --out-dir OUT --cache eval/entities/linking_cache.json
```

- **Mentions.** Mentions of the gold types are matched one-to-one: each gold
  entity, in page order, takes the first unmatched mention that matches it.
  - **exact**: the same surface after NFC, whitespace collapsing and
    casefolding.
  - **fuzzy**: on the search key (abbreviations expanded, u/v and i/j folded),
    the keys are equal, one contains the other (4 or more characters), or
    their CER is at most 1/3. "Roys mͣrcadigas" finds *mͣrcadigas*.
  - **spurious**: the mentions of those types left unmatched.
- **Links.** Links are scored per gold entity through its fuzzy match, as in
  the linking benchmark: correct, wrong, abstained or missed. Dates are not
  linked. GeoNames is not asked, since it needs a username.

### Result (2026-11-24, evaluation VM, live Wikidata and VIAF)

Checkout 0d1fac2, no chat key set. 58 distinct authority requests were made
(Wikidata 33, VIAF 25), and none failed.

| Page | Gold | OCR CER | Gate, OCR / ref. | Mentions exact / fuzzy, OCR | ref. | Spurious, OCR / ref. | Links c / w / a / m, OCR | ref. |
|---|---|---|---|---|---|---|---|---|
| old-french | 0 | 6.5% | HIGH / HIGH | – | – | 0 / 0 | – | – |
| latin-abaton | 0 | 6.6% | HIGH / HIGH | – | – | 0 / 0 | – | – |
| latin-apocalypse | 2 | 18.0% | OK / HIGH | 0 / 0 | 0 / 1 | 1 / 0 | 0 / 0 / 0 / 2 | 0 / 0 / 1 / 1 |
| middle-french-antitus | 5 | 4.1% | HIGH / HIGH | 1 / 3 | 1 / 3 | 0 / 0 | 2 / 0 / 0 / 2 | 2 / 0 / 0 / 2 |
| old-french-verse | 7 | 7.2% | HIGH / HIGH | 1 / 2 | 3 / 5 | 0 / 0 | 2 / 0 / 0 / 5 | 5 / 0 / 0 / 2 |
| **total** | **14** | | | **2 / 5** | **4 / 9** | **1 / 0** | **4 / 0 / 0 / 9** (n=13) | **7 / 0 / 1 / 5** (n=13) |

The CERs and gate verdicts match the recogniser and gate runs
(`scripts/eval_vm/README.md`, 3c and 3d). On the reference text the links
reproduce the linking benchmark: 7 / 0 / 1 / 5. No link is wrong in either
mode.

Where each gold entity is lost:

- **old-french, latin-abaton.** The gold has no named entity on either page,
  so 0 mentions is right. A "MENTION_RECALL_LOW" warning on those pages is a
  false alarm.
- **Lost to OCR** (found on the reference, not on the OCR text):
  - *iohannis*: read "R Rian geliste iohannisi", so the trigger word
    *euangelistæ* is gone. The misread "R Rian" is the one spurious
    mention.
  - *Caldꝰ*, and two occurrences of *mͣrcadigas*: the segmented engine writes
    a versal initial apart from its word ("C aldꝰ li roys", "M arcadigas").
    It does so at the start of 41 of the 70 lines on old-french-verse and 18
    of the 28 on latin-abaton; the references join them. Joining them in the
    OCR text recovers both names here (links 2 -> 4 correct) and lowers CER
    from 7.2% to 5.4% and from 6.6% to 5.5%. A plain regex also joins two
    misreads on latin-apocalypse ("R Rian"), so a fix would need the line
    geometry. Not changed.
- **Lost to linking**: *iohannis* on the reference text is extracted as
  "euangelistæ iohannis" but abstained: no candidate (see "Tried, not
  shipped").
- **Lost to extraction**, on the reference text as well: *hyerusalem*,
  *Ame de monfalcon*, *Antitus*, *Ynabele*, *despaigne* (see "Tried, not
  shipped").

Five pages and one annotator, with 14 gold entities on three pages: this
locates the losses, but it is not a corpus-level score.

