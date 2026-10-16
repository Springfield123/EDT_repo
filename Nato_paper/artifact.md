# CSO symposium paper template distillation

## Reference authority

- User-supplied legacy template: `/Users/huha/Documents/edt_repo/Nato_paper/Paper_template.doc`
- Read-only working conversion used for authoring: `/private/tmp/nato_paper_review/template/Paper_template.docx`
- Legacy template SHA-256: `511c7e8c31097c3ab524876e624c0c060ea211e2a418af019aead95276dbdc5c`
- Retained DOCX SHA-256: `141e21d531814eaf67b458c316912374135b2fe29f69a83675de04bad406ba59`
- Control hierarchy: the 2026 paper instruction governs content, length, and classification; the retained DOCX governs visual design; the sample paper is supporting visual evidence only.

## Layout contract

- A4 portrait, one section, one column.
- Page size: 11906 x 16838 twips (8.27 x 11.69 in).
- Margins: 0.866 in left/right, 1.299 in top, 1.063 in bottom; 0.5 in header; 0.788 in footer.
- Text area: approximately 6.536 x 9.331 in.
- Different first, odd, and even headers/footers are retained.
- Maximum main-paper length: five pages, with references excluded. The output therefore uses five main pages plus a sixth references page.

## Typography and component contract

- Body: `STO Text`, Times New Roman 11 pt, justified.
- Title: `STO Paper Title`, Times New Roman bold 15 pt, centered, maroon `#800000`.
- Author: `STO Author Name`, Times New Roman bold 12 pt, centered.
- Address and e-mail: `STO Author Address` / `STO email address`, Times New Roman 11 pt, centered.
- Abstract: `STO Abstract Title` and `STO Abstract Text`; title and text italic.
- First-level heading: `STO First Level Heading`, Times New Roman bold 13 pt.
- Second-level heading: `STO Second Level Heading`, Times New Roman bold 12 pt.
- Figure/table captions: `STO Figure Caption` / `STO Table Caption`, Arial bold 9 pt, centered.
- References: `STO Reference Text`, Times New Roman 11 pt with hanging indent.
- Figures are centered in `STO Figure` paragraphs and inserted as inline drawings to reduce Word/LibreOffice mismatch.

## Editable slots

- Document properties: title, subject/report identifier, author, category/paper identifier, comments/classification, keywords/release marking.
- First-page title, author, affiliation, country, e-mail, abstract, and body.
- Main body: numbered sections required by the instruction sheet: Introduction, Rationale, Description, Results and Operational Implications, and Conclusions.
- Figures and table: two existing repository figures and one seven-row prospective-validation table.
- References page.
- Header/footer cached text may be patched only to make the required classification and symposium identifier visible in renderers that do not refresh fields.

## Preserve-only structures

- Page geometry and section settings.
- Header/footer relationships, NATO/STO logo asset, horizontal-rule drawings, and alternating odd/even placement.
- Template style definitions and numbering definitions.
- Existing PAGE fields and first-page/odd/even behavior.
- The user-supplied `.doc` is never modified.

## Intentional deviations and assumptions

- The template's instructional pages are replaced by the symposium manuscript.
- The template's 12 pt after-paragraph body spacing is reduced locally, without changing the 11 pt body font, to meet the five-page main-text limit while retaining readable paragraph rhythm.
- The instruction requires the exact visible marking `NATO UNCLASSIFIED/Releasable to STEP, SGP.` in headers and footers. Template metadata fields and cached field text are set accordingly.
- No assigned publication report number or paper number was supplied. The footer uses `IST-SET-226` as the paper/category identifier and `STO-MP-IST-SET-226` as the report identifier rather than inventing a submission number.
- Author affiliations are normalized from the accepted abstract: Hang Hu and Hsu Kiang Ooi at the National Research Council of Canada; Anguang Hu at Defence Research and Development Canada. The supplied corresponding e-mail is retained.
- The retained DOCX is a temporary LibreOffice conversion of a legacy `.doc`; header/footer assets are preserved, but exact binary-package identity with the `.doc` is not possible.

## Content and evidence contract

- The abstract is adopted from the accepted symposium abstract (`Conference_abstract.pdf`), condensed to the template's one-paragraph slot. Its exploratory numbers (`N = 4-10`, bands near `1.35 us` and `3.70 us`) are superseded by the corrected sweep and replaced with the current pipeline output.

- Results are identified as literature-anchored model predictions, not measurements or calibrated field sensitivity.
- The implemented convention is explicit: `N` is the XY8 block count, `N_pi = 8N`, `tau` is half the interpulse spacing, and the primary resonance is `tau = 1/(4f)`.
- The paper reports the canonical 200 kHz optimum (`tau = 1.25 us`, `N = 20`), the 10% depth window (`N = 10-34`, both edges resolved because `N = 9` and `N = 35` are sampled and lie outside it), the `N = 16` Pareto alternative, ablation results, frequency portability, uncertainty/regret results, the hardware-to-control rule, and the unmeasured seven-setting prospective protocol.
- Alternative pulse-sequence claims are excluded because the present scalar instantaneous-pulse model cannot resolve phase cycling.
- The symposium paper is written for policy makers, military personnel, and technical specialists. The Introduction explicitly connects the contribution to the IST-SET-226 aim of bridging those perspectives, while the abstract, model description, figure caption, and tables translate symbolic metrics into plain-language decisions.
- The main text explains the sensitivity score as a relative ranking tool and deliberately omits the standalone implementation equation and coherence exponents. Those details remain reproducible in the repository artifacts and source code rather than competing with the paper's operational message.

## Verification contract

- Render the final DOCX with `render_docx.py --emit_pdf`.
- Inspect every rendered page at full-page and 100% detail.
- Confirm five main pages plus one reference page, no clipped or overlapping text, readable figures/tables, and exact classification in every header and footer.
- Run section, style, heading, image, field, and accessibility audits.
- Compare final package inventory and recurring header/footer assets with the retained template.
