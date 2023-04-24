# Temporal Fusion Transformer (TFT)

Discuss package requirements, optional numba, cuda support, etc. (test_TFT_predictor for notes)

Run files using syntax
```
python3 -m models.TFT.create_train_TFT
```

Requires call of
```
TFT_Predictor.initialise_forecasting(tau)
```
to enter forecasting mode (enable successive inference)