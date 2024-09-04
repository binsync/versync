# versync
An experimental project for transfering symbols across srtripped binary versions using machine learning and other
binary similarity techniques.

## Planned Single Function Workflow
```python
# 1. Extract ACFG features
# 2. Extract ACFG disassembly
# 3. Preprocess ACFG data tuple for embedding
cfg_feature_data = extract_cfg_features("path/to/binary", "func_name")

# 4. Generate embedding
cfg_embedding = generate_embedding(cfg_feature_data)
```

## Planned Full Binary Workflow
```python
# symboled binary
bin_1_embeddings = {
   addr: extract_embedding(cfg_feature_data) for addr, cfg_feature_data in cfg_func_features("path/to/binary1").items()
}
# stripped binary
bin_2_embeddings = {
   addr: extract_embedding(cfg_feature_data) for addr, cfg_feature_data in cfg_func_features("path/to/binary2").items()
}

recover_symbols(bin_1_embeddings, bin_2_embeddings)
# output:
# {addr: (symbol, confidence), addr2: (symbol2, confidence2), ...}
```

## TODO
- [ ] Fix the ACFG extraction for IDA Pro
- [ ] Get the embeddings produced in the test_model function 