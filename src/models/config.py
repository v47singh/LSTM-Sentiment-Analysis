model_config = {
    'clip': 50,
    'lr': 1.0,
    'device': 'cuda:0',
    'n_layers': 1,
    'dropout': 0.5,
    'n_epochs': 200,
    'MAX_LENGTH': 30,
    'batch_size': 64,
    'hidden_dim': 300,
    'dataset': 'sst1',
    'use_attn?': False,
    'embedding_dim': 300,
    'code': 'bilstm_pool1d_no_attn_traditional',
    'operation': 'pool1d',  # attn can't be used with pooling
    'data_dir': 'data/processed',
    'vocab_path': 'data/processed/vocab.npy',
    'filtered_emb_path': 'data/processed/english_w2v_filtered.hd5'
}
