# Source of the set 1 report

`../anafora1.pdf` is generated from `report.html`.

- `report.html` — the report text (Greek), derivations, tables and layout
- `figures/` — the figures of the original report that are still valid (`.jpg`)
  and the plots produced by the corrected scripts in `../exercise1_1`,
  `../exercise1_3`, `../exercise1_4` and `../exercise1_5` (`.png`, run with
  GNU Octave)
- `build.js` — renders the HTML to the PDF

To rebuild after editing the text (needs Node.js with Playwright and Chromium):

```
node build.js
```

The original report submitted in 2020 is in the git history of the PDF.
