Improved Text
==============================

## How to run the code

- Modify hyperparams and other settings (device, lr, paths etc.) `src/models/config.py`
- Run `python src/process_raw.py` to preprocess data
- Run `python src/train.py` to train only on claims. To train with evidences, run `python src_evidence/train.py`

The above commands will generate confusion matrices inside the `reports/figures` folder and save models inside `saved_models` folder
