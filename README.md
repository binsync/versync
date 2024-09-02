# versync
An experimental project for transfering symbols across srtripped binary versions

## Planned Workflow
```python
# 1. Extract ACFG features
# 2. Extract ACFG disassembly
# 3. Preprocess ACFG data tuple for embedding
cfg_feature_data = extract_cfg_features("path/to/binary", "func_name")

# 4. Generate embedding
cfg_embedding = generate_embedding(cfg_feature_data)
```