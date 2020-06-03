import os
import math
import torch
import random
import pickle
import numpy as np
from torch import nn, optim
from datetime import datetime
from collections import Counter
from utils.data import preprocess
from models.lstm import LSTMClassifier
from models.config import model_config as conf

from tensorboardX import SummaryWriter


writer = SummaryWriter()

# Preprocessing
labels_dict = {0: 'V.Neg', 1: 'Neg.', 2: 'Neutral', 3: 'Pos.', 4: 'V.Pos.'}
# labels_dict = {0: 'comp', 1: 'politics', 2: 'rec', 3: 'religion'}

vocab = preprocess.build_vocab(conf)

embedding_wts = preprocess.generate_word_embeddings(vocab, conf) \
    if not os.path.exists(conf['filtered_emb_path']) \
    else preprocess.load_word_embeddings(conf)

x_val = []
y_val = []
y_train = []
x_train = []

train_pairs = preprocess.read_pairs(mode='train', config=conf)
for pair in train_pairs:
    x_train.append(pair[0])
    y_train.append(pair[1])

val_pairs = preprocess.read_pairs(mode='test', config=conf)
for pair in val_pairs:
    x_val.append(pair[0])
    y_val.append(pair[1])

print('Train and Test Label distribution respectively:')
print(Counter(y_train), Counter(y_val))

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = LSTMClassifier(conf, embedding_wts, n_lables=len(labels_dict))
model = model.to(device)
criterion = nn.CrossEntropyLoss(reduction='sum')
optimizer = optim.Adadelta(model.parameters(), lr=conf['lr'], weight_decay=1e-5)

best_f1 = 0
for e in range(conf['n_epochs']):
    losses = []
    all_train_predictions = np.array([])
    all_train_targets = np.array(y_train)
    for iter in range(0, len(x_train), conf['batch_size']):
        input_seq, input_lengths = preprocess.btmcd(
            vocab, x_train[iter: iter + conf['batch_size']])
        targets = torch.tensor(y_train[iter: iter + conf['batch_size']])

        input_seq = input_seq.to(device)
        input_lengths = input_lengths.to(device)
        targets = targets.to(device)

        model.zero_grad()
        optimizer.zero_grad()

        predictions, aux_loss = model(input_seq, input_lengths)
        predictions = predictions.to(device)

        loss = criterion(predictions, targets)
        loss += aux_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), conf['clip'])
        optimizer.step()

        writer.add_scalar('data/train_loss', loss.item(), iter)
        losses.append(loss.item())

        predictions = torch.argmax(predictions, dim=1)
        predictions = np.array(predictions.cpu())
        all_train_predictions = np.concatenate((all_train_predictions,
                                                predictions))
    print('{}>> Epoch: {} | Mean train loss: {}'.format(datetime.now().time(),
                                                        e, np.mean(losses)))
    performance = preprocess.evaluate(all_train_targets, all_train_predictions)
    print('Train A: {acc} | P: {precision} | R: {recall} | F: {f1}'.format(
        **performance))
    # evaluate
    with torch.no_grad():
        all_predictions = np.array([])
        all_targets = np.array(y_val)
        for iter in range(0, len(x_val), conf['batch_size']):
            input_seq, input_lengths = preprocess.btmcd(
                vocab, x_val[iter: iter + conf['batch_size']])

            input_seq = input_seq.to(device)
            input_lengths = input_lengths.to(device)
            pred_logits, _ = model(input_seq, input_lengths)
            predictions = torch.argmax(pred_logits, dim=1)

            # evaluate on cpu
            predictions = np.array(predictions.cpu())

            all_predictions = np.concatenate((all_predictions, predictions))

        # Get results
        preprocess.plot_confusion_matrix(all_targets, all_predictions,
                                         classes=list(labels_dict.keys()),
                                         epoch=e, model_code=conf['code'])
        performance = preprocess.evaluate(all_targets, all_predictions)
        writer.add_scalars('metrics/performance', performance, iter)
        print('Test A: {acc} | P: {precision} | R: {recall} | F: {f1}\n\n'.format(**performance))
        if performance['f1'] > best_f1:
            best_f1 = performance['f1']
            # save model and results
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
                }, 'saved_models/{}_{}_best_model.pth'.format(
                    conf['code'], conf['operation']))

            with open('saved_models/{}_{}_best_performance.pkl'.format(
                    conf['code'], conf['operation']), 'wb') as f:
                pickle.dump(performance, f)
writer.close()
