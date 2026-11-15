# Dataset 1 (Weather) – COMP5434

    python train_predict.py <data_dir with train.csv/test.csv> <out_dir>

Pure numpy/pandas + sklearn (only for CV split & F1 metric). Runtime ~40 s on 1 CPU core.
- features.py : cleaning (Humidity/Precip >100 flagged + clipped), one-hot, interactions
- gbdt.py     : from-scratch histogram multiclass gradient boosting (Newton step, L2 reg)
- mlp.py      : from-scratch MLP (Adam, dropout, label smoothing, cosine LR)
- train_predict.py : 5-fold OOF, blend weight + class-weight search for Macro-F1, 3-seed refit, writes submission
Seeds are fixed, so results are reproducible.
