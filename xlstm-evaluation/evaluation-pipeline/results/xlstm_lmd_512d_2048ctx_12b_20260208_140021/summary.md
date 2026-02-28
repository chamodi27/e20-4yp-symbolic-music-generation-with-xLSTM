# Perplexity Evaluation Summary

**Model**: xlstm_lmd_512d_4096ctx_12b
**Date**: 2026-02-08 14:07

## Model Configuration
- Embedding Dim: 512
- Num Blocks: 12
- Training Context: 2048
- Vocab Size: 675

## Best Checkpoint
- **Checkpoint**: checkpoint-66000
- **PPL at Training Context**: 1.7873

## Checkpoint Selection Results
| checkpoint        |   step |   context_length |   perplexity |
|:------------------|-------:|-----------------:|-------------:|
| checkpoint-2000   |   2000 |             2048 |      4.66536 |
| checkpoint-34000  |  34000 |             2048 |      1.89257 |
| checkpoint-66000  |  66000 |             2048 |      1.78732 |
| checkpoint-98000  |  98000 |             2048 |      1.83482 |
| checkpoint-130000 | 130000 |             2048 |      2.19309 |
| checkpoint-158760 | 158760 |             2048 |      2.50464 |

## Test Results (All Context Lengths)
| checkpoint       |   step |   context_length |   perplexity |
|:-----------------|-------:|-----------------:|-------------:|
| checkpoint-66000 |  66000 |             1024 |      1.91782 |
| checkpoint-66000 |  66000 |             2048 |      1.77647 |
| checkpoint-66000 |  66000 |             3072 |      1.77545 |
| checkpoint-66000 |  66000 |             4096 |      1.88193 |
| checkpoint-66000 |  66000 |             5120 |      2.08539 |
| checkpoint-66000 |  66000 |            10240 |      3.55022 |
