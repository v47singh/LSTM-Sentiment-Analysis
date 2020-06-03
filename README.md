Improved Sentiment Analysis using LSTMs
==============================

## How to run the code

- Modify hyperparams and other settings (device, lr, paths etc.) `src/models/config.py`
- Run `python src/process_raw.py` to preprocess data
- Run `python src/train.py` to train only on claims. 

The above commands will generate confusion matrices inside the `reports/figures` folder and save models inside `saved_models` folder

Please refer to the [report](reports/FinalReport.pdf) for more details about the classification network.
