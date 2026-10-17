# Hearing front-end fit to infant behavioural pure-tone thresholds

`python3 fit.py` reads `data.csv`, writes `fit.json` and `fig_hearing_fit.png`.
Quantity: infant threshold minus adult threshold (dB), behavioural tasks. Low = 250-500 Hz, high = 4 kHz (8 kHz noted separately).

## Data actually obtained (see data.csv for locators)
- Olsho et al. 1988 (full text read): 3 mo 23 dB at 500 Hz, 19 dB at 4 kHz, ~30 dB at 8 kHz; 6 mo ~15 dB at 250 Hz, <10 dB at 8 kHz; 6+12 mo pooled 10-15 dB, larger at low frequency. Values are stated in text, not read off Fig. 4.
- Werner 2007 (full text read): 1 mo 4 kHz ~35; 3 mo 4 kHz 30 (conflicts with Olsho's 19); 6 mo >2 kHz ~15; low frequencies ~20 dB poorer thereafter (age unspecified).
- Werner & Gillenwater 1990: 45 dB at 500 Hz, 35 at 4 kHz (1 mo). Original not readable; 45 dB comes only from Werner's course notes, so it is excluded from all fits; 1 month uses 4 kHz only.
- Not obtainable: Trehub et al. 1980, Sinnott et al. 1983, Berg & Smith 1983 (no accessible text or numbers); dropped. PubMed/PMC/Crossref/Europe PMC were blocked or rate-limited.

## Missing data
No behavioural newborn (0 mo) adult-referenced values; no 12-month-only or second-year values (12 mo is a pooled 6+12 mo range, midpoint 12.5); nothing between 12 months and ~4 years. Fitted values at 0 and 24 months are extrapolation (exponential gives 61 dB at 0 mo, not credible; 24 mo is just the plateau). Four points, three parameters: shape is weakly constrained.

## Results (primary set)
Exponential 36.1*exp(-t/2.10)+12.6 on mean(low,high), 1 month 4 kHz only: 1 mo 35.0, 3 mo 21.2, 6 mo 14.6, 12 mo 12.7 dB (24 mo ~12.6, assumption of plateau; 0 mo extrapolated).
Current anchors (interpolated): 1 mo 23.3, 3 mo 20, 6 mo 12, 12 mo 6, 24 mo 3, 0 mo 25.
Fit minus current: 1 mo +16.6, 3 mo +1.3, 6 mo +2.3, 12 mo +6.9, 24 mo +9.9 (extrapolated). Alternative set (Werner 3-mo 4 kHz = 30): 3 mo 26.1 (+6.1), 6 mo 16.8, 12 mo 12.3.

Treble: high minus low = -10 (1 mo), -4 (3 mo), 0 (6 mo); with Olsho's 8 kHz, +7 at 3 mo. Behavioural data do not show a larger high-frequency loss at 4 kHz; the current treble cut (12/6/4/1 dB) is not supported by these data. Physiological (ABR/middle-ear) literature does suggest high-frequency immaturity, but that was not used here.
