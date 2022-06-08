# Report-02

## Abstract
This document will summarise more advanced models we've tried:
- Deep Linear Networks.
- XGB Regressor.


## Experiments Docs:
- **Deep Linear Networks**:
  - Initially set all hidden widths to be the same, then added option
  different widths. At this point it seems same width is better,
  but need to explore a lot more options to be sure - only option 
  checked is growing and then shrinking hidden size. Perhaps other
  way round will do better.
  - Right now, best result is 11.53 MSE on Val-set, with hidden widths 
  all set to 150, depth 5 (i.e. 4 hidden layers), lr=0.000819, dropout=
  0.528615, weight decay=0.000016. Can be seen [here](https://www.comet.ml/nadbag98/template-tests/48d4a5fc4be34be4875f139707498237).
  
---------
## Leads:
  - **More work on Linear Nets**:
    - Run more comprehensive experiments on server, checking many
    more combinations of parameters.
  - **Random Forest Regressors**:
    - Implement and run experiments.
