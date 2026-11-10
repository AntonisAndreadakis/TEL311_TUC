// Renders report.html to the report PDF.  Usage: node build.js
// Needs Playwright with Chromium (npm install playwright).
const path = require('path');
const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto('file://' + path.join(__dirname, 'report.html'));
  await page.pdf({
    path: path.join(__dirname, '..', '2013030059_Andreadakis.pdf'),
    format: 'A4',
    printBackground: true,
    preferCSSPageSize: true,
  });
  await browser.close();
})();
