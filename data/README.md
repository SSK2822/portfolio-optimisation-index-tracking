# Data

Weekly CSV files (comma separated, no header), made by `scripts/prepare_data.py`.

```
bcst/<DATASET>/asset_returns.csv    weeks x assets, weekly returns
bcst/<DATASET>/index_returns.csv    index returns
bcst/<DATASET>/risk_free.csv        weekly risk-free rate
updated/<DATASET>/asset_prices.csv  weeks x assets, weekly prices
updated/<DATASET>/index_prices.csv  index level
updated/<DATASET>/risk_free.csv     weekly risk-free rate
```

- **BCST-Library:** DJIA, NASDAQ100, FTSE100, S&P500. Bruni, R., Cesarone, F.,
  Scozzari, A., and Tardella, F. (2016). *Data in Brief*, 8, 858-862.
- **Updated Library:** HDAX, FTSE100, HSCI, S&P500 (as used in Leung et al., 2022a)
  - `SP500U` = Updated S&P500 (separate from the BCST S&P500)
  - Its index series has one fewer weekly price than the asset prices
