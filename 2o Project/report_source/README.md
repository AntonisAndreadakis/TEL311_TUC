# Source of the set 2 report

`../2013030059_Andreadakis.pdf` is generated from `report.html`.

- `report.html` — the report text (Greek), tables and layout
- `figures/orig-*.jpg` — the figures, derivations and output screenshots of the
  original report that are still valid (extracted from the 2020 PDF)
- `figures/*.png` — the new plots, produced by the corrected scripts
  (`../exercise2_4/ex_2_4.m`, `../exercise2_5/ex_2_5.m`, `../exercise2_6/ex_2_6a.m`,
  `../exercise2_6/ex_2_6b.m`, `../exercise2_7/svm_exercise1.m` – `svm_exercise3.m`)
  run in GNU Octave; for exercise 2.7 Octave's `qp` solver was used in place of
  MATLAB's `quadprog`
- `build.js` — renders the HTML to the PDF

To rebuild after editing the text (needs Node.js with Playwright and Chromium):

```
node build.js
```

The original report submitted in 2020 is in the git history of the PDF.
