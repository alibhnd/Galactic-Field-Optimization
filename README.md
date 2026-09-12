# Engineering data

The engineering case-study script expects the following files in this folder:

- `Carbonation.xlsx`
- `Geopolymer.xlsx`

The files are intentionally not included in the repository unless their
redistribution is permitted.

## Carbonation-depth dataset

Expected worksheet: `Carbonation`

The script resolves aliases for the following 14 predictors:

1. Cement
2. Fly ash
3. Water
4. Water-to-binder ratio
5. Coarse aggregate
6. Recycled aggregate
7. Equivalent water absorption of coarse aggregate
8. Fine aggregate
9. Superplasticizer
10. 28-day compressive strength
11. CO2 concentration
12. Exposure time
13. Temperature
14. Relative humidity

Target: carbonation depth.

The reported study uses 819 observations. Missing Temperature and Relative
Humidity values are replaced by their respective column means. No other
missing predictor or target value is silently imputed.

## Recycled geopolymer concrete dataset

Expected worksheet: `CS_Recycled Geopolymer`

The script resolves aliases for these 15 predictors:

1. FA
2. GGBS
3. MK
4. NCA
5. NWA
6. RCA
7. RWA
8. NFA
9. NH
10. NHC
11. NS
12. CT
13. HCD
14. SP
15. Age

Target: compressive strength.

The reported study uses 373 observations and does not apply the carbonation
mean-imputation rule to this dataset.

If your source files use different column labels, update the alias dictionaries
near the top of `gfo_engineering_svr.py`.
