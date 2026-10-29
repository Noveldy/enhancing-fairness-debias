# Air Quality, FSDG Fourier value functions vs real observed data

| Feature | KS (mean ± SD) | Wasserstein (mean ± SD) | real min-max | synthetic min-max |
|---|---|---|---|---|
| CO(GT) (mg/m³) | 0.101 ± 0.004 | 0.315 ± 0.006 | 0.1 – 11.9 | -4.1 – 7.7 |
| PT08.S1(CO) (sensor response) | 0.077 ± 0.002 | 35.7 ± 0.7 | 647 – 2040 | 153 – 2092 |
| C6H6(GT) (µg/m³) | 0.106 ± 0.003 | 1.75 ± 0.03 | 0.1 – 63.7 | -22.8 – 41.4 |
| PT08.S2(NMHC) (sensor response) | 0.053 ± 0.003 | 31.7 ± 1.1 | 383 – 2214 | -292 – 2002 |
| NOx(GT) (ppb) | 0.136 ± 0.004 | 63.5 ± 1.3 | 2 – 1479 | -540 – 1107 |
| PT08.S3(NOx) (sensor response) | 0.056 ± 0.005 | 33.0 ± 1.5 | 322 – 2683 | -191 – 1866 |
| NO2(GT) (µg/m³) | 0.046 ± 0.003 | 4.90 ± 0.17 | 2 – 340 | -77 – 301 |
| PT08.S4(NO2) (sensor response) | 0.024 ± 0.003 | 18.3 ± 1.1 | 551 – 2775 | 36 – 2664 |
| PT08.S5(O3) (sensor response) | 0.068 ± 0.006 | 52.0 ± 3.1 | 221 – 2523 | -571 – 2663 |
| T (°C) | 0.020 ± 0.003 | 0.305 ± 0.040 | -1.9 – 44.6 | -7.6 – 46.7 |
| RH (%) | 0.036 ± 0.002 | 1.20 ± 0.03 | 9.2 – 88.7 | -17.7 – 115.3 |
| AH (absolute humidity, unit not stated by UCI) | 0.032 ± 0.003 | 0.0225 ± 0.0015 | 0.1847 – 2.2310 | -0.3871 – 2.4156 |
| Mean over the 12 columns | 0.063 ± 0.001 | – | – | – |

KS and Wasserstein distance between each synthetic column (9,357 values per seed) and the real observed values of that column (imputed cells excluded); mean ± SD over seeds [0, 1, 2, 3, 4]. Wasserstein distances are in the column's unit and are not averaged across columns. Synthetic min-max over all 5 seeds.
