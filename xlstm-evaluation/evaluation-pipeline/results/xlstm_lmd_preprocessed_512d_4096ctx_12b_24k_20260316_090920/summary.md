# Perplexity Evaluation Summary

**Model**: xlstm_lmd_preprocessed_512d_4096ctx_12b_24k
**Date**: 2026-03-16 09:09

## Model Configuration
- Embedding Dim: 512
- Num Blocks: 12
- Training Context: 4096
- Vocab Size: 413

## Best Checkpoint
- **Checkpoint**: checkpoint-38000-last
- **PPL at Training Context**: 1.4808

## Checkpoint Selection Results
| checkpoint            |   step |   context_length |   perplexity |
|:----------------------|-------:|-----------------:|-------------:|
| checkpoint-2000       |   2000 |             4096 |      2.2863  |
| checkpoint-14000      |  14000 |             4096 |      1.57603 |
| checkpoint-26000      |  26000 |             4096 |      1.50503 |
| checkpoint-38000-last |  38000 |             4096 |      1.48082 |
| checkpoint-50000      |  50000 |             4096 |      1.52027 |
| checkpoint-65080      |  65080 |             4096 |      1.64242 |

## Test Results (All Context Lengths)
| checkpoint            |   step |   context_length |   perplexity |
|:----------------------|-------:|-----------------:|-------------:|
| checkpoint-38000-last |  38000 |             1024 |      1.67181 |
| checkpoint-38000-last |  38000 |             2048 |      1.5561  |
| checkpoint-38000-last |  38000 |             3072 |      1.50971 |
| checkpoint-38000-last |  38000 |             4096 |      1.48464 |
| checkpoint-38000-last |  38000 |             5120 |      1.4953  |
| checkpoint-38000-last |  38000 |             8192 |      2.33925 |
| checkpoint-38000-last |  38000 |            10240 |      3.5774  |
