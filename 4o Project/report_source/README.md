# Source of the set 4 report

`../2013030059_Andreadakis.pdf` is generated from `report.html`.

- `report.html` — the report text (Greek), tables and layout
- `figures/` — the images of the original report that are still valid
  (`ex1-*`, `ex2-cross-entropy`, `ex2-derivation-a`, `ex3-*`) and the two new
  plots of exercise 2 (`ex2-loss.png`, `ex2-boundary.png`), produced from the
  output of `../exercise4_2/matlab/logisticRegression.m` (run in GNU Octave,
  `randn('state',1)`) and `../exercise4_2/python/logisticRegression.py`
  (`np.random.seed(0)`)
- `build.js` — renders the HTML to the PDF

To rebuild after editing the text (needs Node.js with Playwright and Chromium):

```
node build.js
```

The original report submitted in 2020 is in the git history of the PDF.
