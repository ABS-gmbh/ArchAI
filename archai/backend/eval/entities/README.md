# Entity extraction benchmark

**On held-out reference text, names that have an authority record are linked correctly 2 times in 153, never wrongly, and most are never extracted; the five development pages (7 of 9 linked) badly overstated it** (2026-11-24, "Held-out name set" below).

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

`linking_cache.json` (about 380 kB) holds the 45 responses recorded on
2026-11-23 with `--record`, and 10 added on 2026-11-24 for the queries the
name trimming makes (the 45 unchanged). `--record` uses the app's
User-Agent, sends at most one request a second, and stops on a 429 or 403. They are trimmed to what the
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

**With versal initials joined** (`OCR_JOIN_DETACHED_INITIALS`, the default
since 4f68977; same VM, authority responses replayed from the run above):
old-french-verse goes from CER 7.2% to 5.4%, and on its OCR text *Caldꝰ*
and a second *mͣrcadigas* are found. Over all pages, on the OCR text:

| | Mentions exact / fuzzy | Spurious | Links c / w / a / m (n=13) |
|---|---|---|---|
| before | 2 / 5 | 1 | 4 / 0 / 0 / 9 |
| after | 3 / 7 | 1 | 6 / 0 / 0 / 7 |
| reference text | 4 / 9 | 0 | 7 / 0 / 1 / 5 |

Five pages and one annotator, with 14 gold entities on three pages: this
locates the losses, but it is not a corpus-level score.


## Held-out name set (B-dev, B-test)

**On held-out reference text, names that have an authority record are linked correctly 2 times in 153, never wrongly, and most are never extracted; the five development pages (7 of 9 linked) badly overstated it** (2026-11-24).

The five pages above set the extraction rules and the linking thresholds, so
they cannot judge a change to either. `heldout_pages.json` and `heldout_gold.json`
are a second set, built 2026-11-24 from the reference transcriptions of the
CATMuS Medieval **test** manuscripts (Hugging Face `CATMuS/medieval`, CC-BY 4.0,
revision e119659; text only). None of these manuscripts is in the dataset's
train or dev split, and none was used to build anything in this repository.

- **Pages.** Each pseudo-page is a block of up to 30 consecutive main-text
  lines of one manuscript, with the line ids (`<parquet file>#<row>`) it came
  from. Blocks were chosen to contain names, by a count of capitalised words
  and of titles and roles (rex, comes, sanctus, roy, conte, don, rey...), then
  read. They were not chosen by running the extractor, which would favour pages
  it already handles.
- **Split.** By manuscript: no manuscript is in both halves. **B-dev** may be
  read case by case to find what loses names. **B-test** is used once, to
  judge a change chosen on B-dev, and its errors are not inspected one by one.

### Annotation guidelines

Written before any page was annotated.

1. **What is a mention.** A name of:
   - a **person**: individuals, saints and biblical figures named in the
     text, including a name used in a genitive or other inflected form;
   - a **place**: towns, regions, countries, rivers, churches and abbeys
     named as places;
   - a **work**: a book or text named by its title.
   Not mentions: titles and roles without a name ("le roy", "abbas"),
   peoples and adjectives ("Franci", "françois"), God and Christ as divine
   names ("deus", "dieu", "ihesu crist"), and dates. Feast days formed on a
   saint's name ("la saint iehan") are not mentions of the saint.
2. **The span** is the name as written, abbreviations and all, including
   the parts that belong to it ("iohannes de monte", "saint denis",
   "Robertus dictus Bigot"), without a leading title or role ("rex",
   "messire", "don"). A name split across lines is one mention, with the
   line break in its span. `occurrence` gives the nth occurrence of the same
   surface on the page, when it repeats.
3. **Type**: person, place or work. A saint named as a church or abbey
   ("sancti dyonisii" meaning the abbey) is a place.
4. **Authority.**
   - `linkable` with Wikidata QIDs, and VIAF ids where the Wikidata item
     gives them (P214), when the person, place or work is identifiable from
     the page and has an item. Places get the place's item; GeoNames is not
     used.
   - `none` when no record can be expected: most individuals in charters,
     witnesses, tenants, characters of fiction without an item.
   - `uncertain` when a record may exist but the page does not settle which
     (a common name, an ambiguous place). These are scored as found or
     missed, not as linked or not.
5. **Doubt.** `unsure: true` marks a mention whose reading, span or type the
   annotator is not sure of. Unsure mentions are reported apart and do not
   count against the extractor.
6. **Who.** One annotator (Claude, an AI model), reading diplomatic
   transcriptions without the page images, with no specialist check.
   Authority ids were looked up with Wikidata's search API, and every choice
   is recorded in `how`.

### The set (2026-11-24)

38 pseudo-pages from 22 manuscripts, 1,135 lines:

- **B-dev:** 21 pages, 11 manuscripts. Latin charters and statutes, an
  11th-century office, French narrative and verse, Castilian, Navarrese,
  Occitan and Italian.
- **B-test:** 17 pages, 11 manuscripts. Latin charters, a passio, French
  narrative, Castilian chronicle and treatise, Venetian chronicle, Italian
  and Catalan.

Gold, one annotator, without the images:

| | sure mentions | unsure | linkable | uncertain | none |
|---|---|---|---|---|---|
| B-dev | 152 | 96 | 108 | 83 | 57 |
| B-test | 120 | 67 | 81 | 58 | 48 |

- **Line order.** The dataset's row order is not the reading order of a page:
  lines of a block come out of sequence. A name split across two lines can
  rarely be rejoined, so only names within a line are annotated.
- **Authority ids.** Looked up with wbsearchentities. Six searches found no
  item (Mars, Pan, Arcadia, Solomon, Stephen of Bar, the abbey of
  Saint-Victor). Those mentions are `uncertain`, not filled in from memory.

### Result on the reference text (2026-11-24)

`scripts/benchmark_names_heldout.py`, current code. Authority responses were
recorded live (1,107 requests, none failed) to `heldout_linking_cache.json.gz`
(gzip; SHA-256 of the uncompressed JSON e3f1030bc9533d85e72e5fb76356a2b335b8bbb2477b71bc29ea471492f0e59a), so
reruns are offline. The changes below added the requests their new queries
make (1,538 distinct requests after the spurious-mention rules). Unsure gold mentions are left out. Intervals are 95%,
resampled over manuscripts.

| | gold | found | spurious / extracted | links c / w / a / m |
|---|---|---|---|---|
| five development pages (above) | 14 | 9 (fuzzy) | 0 spurious | 7 / 0 / 1 / 5 (n=13) |
| B-dev, all | 152 | 52 [0.14, 0.51] | 65 / 129 | 23 / 3 / 22 / 90 (n=138) |
| B-dev, linkable only | 91 | 23 | | 1 / 0 / 22 / 68 |
| B-dev, no record (`none`) | 47 | 25 | | 22 / 3 / 0 / 22 |
| B-test, all | 120 | 28 [0.07, 0.44] | 35 / 87 | 18 / 0 / 7 / 87 (n=112) |
| B-test, linkable only | 62 | 8 | | 1 / 0 / 7 / 61 |
| B-test, no record (`none`) | 43 | 17 | | 17 / 0 / 0 / 26 |

By language, found / gold, B-dev and B-test together:

| Language | Found / gold |
|---|---|
| Latin | 56/120 |
| French | 12/42 |
| Castilian | 4/51 |
| Italian | 6/14 |
| Navarrese | 1/4 |
| Occitan | 0/7 |
| Catalan | 0/3 |
| Venetian | 1/31 |

On pages the rules and thresholds were not set on, the extractor finds about
a third of the names. On the five development pages it found about two
thirds. Of the names that have an authority record, 2 of 153 are linked
correctly. None is linked wrongly; the rest are abstained (29) or never
found (129). Of the names with no record (`none`), 39 of the 42 found were
rightly left unlinked; 3, all in B-dev, were linked to something.

These figures rest on 22 manuscripts and one annotator reading without the
images. Many intervals are wide, and single-manuscript languages have none.
No rule or threshold was changed for this measurement. B-test's errors have
not been inspected one by one; it is kept to judge one change chosen on
B-dev.

### Changes measured on B-dev

Each change is its own commit, measured on B-dev and on the five development
pages before it was kept. Development pages, links c / w / a / m: reference
7 / 0 / 1 / 5, end-to-end on the saved OCR text 6 / 0 / 0 / 7, unless said.

| Change | found / gold | spurious / extracted | links c / w / a / m | linkable | none |
|---|---|---|---|---|---|
| baseline | 52 / 152 | 65 / 129 | 23 / 3 / 22 / 90 | 1 / 0 / 22 / 68 | 22 / 3 / 0 / 22 |
| "sainct" a title; "la"/"le"/"les" after "de" a particle | 55 / 152 | 65 / 134 | 23 / 3 / 25 / 87 | 1 / 0 / 25 / 65 | 22 / 3 / 0 / 22 |
| and a sentence point (. ; :) ends a run of capitals | 51 / 152 | 63 / 129 | 19 / 3 / 25 / 91 | 1 / 0 / 25 / 65 | 18 / 3 / 0 / 26 |
| and a person is asked for by name alone; one name needs the context | 51 / 152 | 63 / 129 | 19 / 3 / 25 / 91 | 1 / 0 / 25 / 65 | 18 / 3 / 0 / 26 |
| and rubric sigla, number words, bare titles and feasts are no names | 53 / 152 | 38 / 106 | 19 / 3 / 27 / 89 | 1 / 0 / 27 / 63 | 18 / 3 / 0 / 26 |
| and a byname inside a person's name is not a place of its own | 53 / 152 | 36 / 99 | 22 / 0 / 27 / 89 | 1 / 0 / 27 / 63 | 21 / 0 / 0 / 26 |

- **Lexicon.** Of the 33 capitalised B-dev names never proposed, 2 were lost
  to a missing spelling of a title ("sainct Pol", "sainct Frãcois") and 1 to
  "de la" breaking a run of capitals ("Rommant de la Rose"). All three are
  now found and abstained on. Other spellings of the saint titles were
  checked on B-dev and the development pages and not added, since no name
  was lost for want of them: "sancto" (sc̃o Thome, sc̃o patͥc̃o: found
  through other rules), "san" ("Sã rozi", no gold name), "sanctorum" (no
  name follows), and "sainte", "saincte", "sant", "santa", "sanctum",
  "beato" (absent). The other 30 have no rule that could take them: 22 are
  a lone capitalised word with no title or preposition before it (a lone
  capital mid-line is a gold name 64 times in 227 on B-dev, so no rule
  proposes it), 8 a capitalised given name with a lowercase byname
  ("Thomas de hereforde").
- **Sentence points.** The run of capitals went across ". ; :", making
  "Sancti Marci. In", "Ducentesimo. Sexagesimo", "Dominos Franciscum.
  Iacobum". It now stops there, except at a point after an abbreviated word
  (a combining mark or an abbreviation letter: "Theob̃. Walt̃"). The table
  shows four gold names fewer found: Iacobum and Biã three times. They were
  never found as names: each sat inside a span running across a point into
  the next sentence ("Biã. La", "Biã. Accio", "Dominos Franciscum.
  Iacobum"), and the scoring counted it found because the span overlapped
  the name, and correct because a name with no record that is left unlinked
  scores as correct. That was an artefact of the scoring, not a find. After
  the cut each is a lone capital, which no rule proposes. The spans are now
  the names alone, and they are the baseline the next changes (the query
  trimming and the spurious-mention rules) are measured against.
- **Query trimming** (cause 2). A person's span was sent to Wikidata and
  VIAF as it was: "Sainct Pierre", "beati iohannis obtutibus presentatus",
  "Rey Rodrigo". The query, and the string candidates are compared with, is
  now the name (`_name_query`): titles, roles, rubric words and
  sentence-start words dropped, cut at a sentence mark, only the name kept
  after a title; abbreviation marks are dropped rather than turned into
  spaces ("Rog̾o" was queried as "Rog o"). The span shown is unchanged, and
  the decision trace records the query as `name_query`. B-dev does not move:
  the trimmed names are mostly one given name ("Pierre", "Marc", "iohannis",
  "Rodrigo"), and such a query finds every bearer of the name. Of the 63
  one-name person queries on B-dev, 28 had a top candidate over the 0.80
  threshold (Pierre Abélard 0.81 for "Sainct Pierre", Rodrigo Q376394 0.87
  for "Rey Rodrigo") and were held back only by the 0.15 margin to the
  next; 12 found nothing, 18 scored under 0.80, 5 were stopped earlier. So
  one name alone no longer links: the candidate's description (its label
  and the queried name left out) must have a word in the page around the
  mention. With the margin rule switched off as a stress test (not a
  setting), names with a record would get 14 wrong links without this
  guard and 2 with it (both two-word names, "Theob̃. Walt̃" and a duplicate
  "Pierre Gringoire"); it also turns away the one right link that run
  makes (COLUMBANI). With the margin on, as shipped, it changes nothing on
  B-dev. The 3 wrong links on names with no record are not touched: they
  come from one-word place mentions ("Renni" to Rennington, "Tiper" to
  Tipperary), not from person spans.
- **Spurious mentions** (cause 5). Four rules, all in the mention
  extractor: a rubric siglum, one capital with an abbreviation mark (A̾
  antiphon, R̃ responsory, U̾ versicle, P̾ psalm), neither starts nor continues
  a run of capitals; nor does a spelled-out number (Millesimo, Ducentesimo,
  Tercio); a run made only of titles, roles and function words ("Dominus
  Dux", "Quod Dñs Dux") is dropped; a saint's name right after a feast word
  (festum, festo, festiuitatibus, die, uigilia, ...) is a date, a feast,
  not a person ("in festo sancti michaelis"), and is not propagated to.
  28 spurious mentions are removed: 21 by the sigla rule, 4 bare titles,
  3 feasts; the number words had already been cut apart by the separator
  change, so none of the 28 is theirs alone. 3 are added: the psalm incipit
  "Beati uiri" twice, which the title rule now reaches once "A̾ Beati" no
  longer covers it, and "S Acri" from "A̾ S Acri". No gold name is lost;
  two ("Sancte iohannes", the same reason as "Beati uiri") are newly found
  and abstained on. The 3 wrong links on names with no record are not
  touched.
- **Bynames.** The 3 wrong links on names with no record came from the
  rule that proposes a capitalised word after "de" as a place: in a witness
  list ("Rog̾o de Renni Simone", "Galfrido de Tiper") the byname was a place
  of its own and was linked, "Renni" to Rennington, "Tiper" to Tipperary. A
  place candidate lying inside a person's run of capitals is no longer
  proposed, unless it is an anchored place (lausanne, paris, ...). The
  three wrong links go, as does one spurious place ("Iuifs" in "Roys des
  Iuifs Ihesus"); no gold name is lost, and the development pages are
  unchanged, lausanne still linked. What it gives up: a real place inside
  a byname ("Iehan de Paris" with an unanchored place) is no longer a place
  mention of its own; the person mention still covers it. A run of
  capitals is at most five names, so in a longer witness list a later
  byname still comes out as a place.

Tried on B-dev, not shipped:

- **Letting a dropped title choose among the namesakes** (2026-11-24). When
  the title trimmed off a person's span is a saint's ("sainct", "beati"), a
  ruler's ("rey", "dux") or a church office ("episcopus"), only candidates
  whose label or description says they are of that kind were rivals in the
  margin comparison, the one-name context check still applying and no
  threshold changed. B-dev did not move: 0 links gained, 0 wrong (links
  22 / 0 / 27 / 89 before and after). Of the 34 titled person mentions,
  32 reached the scorer and 14 of those had a candidate of the kind among
  the answers, some only by a word ("Saint-Pierre" in a surname, Cyrus II
  of Persia for "Çiro Rey" at 0.788). Where the right one was found it
  still could not link: Mark the Evangelist (Q31966) for "Sainct Marc"
  and Andrew the Apostle (Q43399) for "Sainct Andre" were never enriched
  (the linker enriches at most 3 Wikidata candidates a mention, in the
  order of a pre-sort that does not know the title), so their type was
  unknown and they were turned away as type-incompatible; Saint George
  (Q48438), enriched, became the only rival for "Sainct George" but scored
  0.770, under the 0.80 threshold. A title could only help with the
  enrichment order changed as well, which this attempt did not try.
