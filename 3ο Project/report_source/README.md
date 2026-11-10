# Source of the set 3 report

`../2013030059_Andreadakis.pdf` is generated from `report.html`.

- `report.html` — the report text (Greek), tables and layout
- `figures/` — the k-means figures of the original report and the GMM / k-means
  plots produced by `../exercise3_3/test_gmms.m`
- `build.js` — renders the HTML to the PDF

To rebuild after editing the text (needs Node.js with Playwright and Chromium):

```
node build.js
```

The original report submitted in 2020 is in the git history of the PDF.
