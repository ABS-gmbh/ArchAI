# Junicode

The second transcription face (`--font-junicode`, after Gentium Plus in the
`font-text` stack): it draws what Gentium Plus lacks.

- **Version:** 2.226
- **Source:** Peter S. Baker's release,
  https://github.com/psb1558/Junicode-font/releases/download/v2.226/Junicode_2.226.zip
  The file is `Junicode/WOFF2/JunicodeVF-Roman.woff2` from that archive,
  unchanged: the variable Roman, weights 300 to 700.
- **Licence:** SIL Open Font License 1.1, in [`OFL.txt`](OFL.txt).

## Why it is here

The Kraken recognisers output characters Gentium Plus 6.200 does not have.
They include combining letters such as U+1DE4 (s above: "boneᷤ",
"ꝓfitableᷤ"), U+1DD1 (ur above) and U+1DE0 (n above), and ℥, ❧ and ⁋. They
also include MUFI private-use characters. With no font for them, a line showed
an empty box, or set one letter in a system face.

Junicode is the medievalists' face, with MUFI coverage. A base letter and its
combining mark that Gentium cannot draw together fall to Junicode as one
cluster, so the mark sits on its letter.

`node scripts/check-font-coverage.mjs` (in `archai/frontend`) lists, for each
recogniser's character set (`scripts/fixtures/recogniser-charsets.json`), what
neither font draws. For CATMuS Medieval, CREMMA Medieval and McCATMuS that is
nothing, except U+F2DA, a MUFI private-use point that Junicode 2.226 does not
have either.

The Italic (1.2 MB) is not bundled: transcriptions are not set in italic.

| File | SHA-256 |
|---|---|
| JunicodeVF-Roman.woff2 | 1731c456638bc75cd84f8b080505a56a7e137997e2a598cef7fe6e1ab02fcf4d |
