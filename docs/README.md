---
layout: home
permalink: index.html
repository-name: e20-4yp-Exploring-xLSTM-for-Long-Term-Structural-Coherence-in-Symbolic-Music-Generation
title: Exploring xLSTM for Long-Term Structural Coherence in Symbolic Music Generation
---

# Exploring xLSTM for Long-Term Structural Coherence in Symbolic Music Generation

#### Team

- E/20/037, Haritha Bandara, [e20037@eng.pdn.ac.lk](mailto:e20037@eng.pdn.ac.lk)
- E/20/363, Yohan Senanayake, [e20363@eng.pdn.ac.lk](mailto:e20363@eng.pdn.ac.lk)
- E/20/365, Chamodi Senaratne, [e20365@eng.pdn.ac.lk](mailto:e20365@eng.pdn.ac.lk)

#### Supervisors

- Dr. Isuru Nawinne, [isurunawinne@eng.pdn.ac.lk](mailto:isurunawinne@eng.pdn.ac.lk)
- Isuri Devindi, [isurid@umd.edu](mailto:isurid@umd.edu)

#### Table of content

1. [Abstract](#abstract)
2. [Related works](#related-works)
3. [Methodology](#methodology)
4. [Experiment Setup and Implementation](#experiment-setup-and-implementation)
5. [Results and Analysis](#results-and-analysis)
6. [Conclusion](#conclusion)
7. [Publications](#publications)
8. [Links](#links)

---

## Abstract

Generating long-form symbolic music with coherent structure remains a significant challenge in AI music generation. While Transformer-based approaches have made notable progress — with models like Museformer introducing specialized attention mechanisms for long music sequences — the dominant paradigm still relies on attention-based architectures. In this work, we explore the Extended Long Short-Term Memory (xLSTM) architecture as an alternative for symbolic music generation, leveraging its recurrent state formulation to achieve efficient, linear-time generation of arbitrarily long sequences with constant memory usage. We train an xLSTM model on the Lakh MIDI Dataset using the REMIGEN encoding and evaluate its generation capabilities across sequence lengths from 2,048 to 12,288 tokens. A key practical contribution is the implementation of a recurrent-state inference method that correctly utilizes xLSTM's intended generation mode, achieving significant speedups over a naive parallel-mode baseline. We analyze the generated output in terms of musical properties, generation performance, and token grammar adherence. We further evaluate the output using objective musical metrics and human listening tests, which indicate meaningful repetition patterns, musically plausible note distributions, and perceptible long-term structural coherence. Our results demonstrate that xLSTM is a viable architecture for long-form symbolic music generation, offering a fundamentally different trade-off compared to Transformer-based approaches.

## Related works

**Symbolic Music Generation**: Symbolic music generation operates on discrete representations such as MIDI and has evolved through several architectural paradigms. Early approaches used RNNs and LSTMs, which struggled with long-range dependencies. The Music Transformer introduced relative attention for MIDI generation, enabling richer long-range modeling but still facing quadratic complexity with sequence length.

**Handling Long Sequences**: Standard full attention has quadratic complexity, which limits its applicability to long music sequences that can span tens of thousands of tokens. Museformer (Yu et al., NeurIPS 2022) addresses this by introducing fine- and coarse-grained attention, where tokens directly attend to structure-related bars at full resolution and summarize other bars at lower cost. This achieves effective complexity between linear and quadratic, enabling full-song generation. Other approaches include the Bar Transformer for hierarchical structure learning and whole-song generation via cascaded diffusion models.

**xLSTM and Modern Recurrent Architectures**: The xLSTM architecture (Beck et al., NeurIPS 2024) revisits the LSTM with exponential gating and matrix memory, introducing two variants: sLSTM (scalar memory with memory mixing) and mLSTM (matrix memory with covariance updates). A key property is its dual formulation — parallel mode for training and recurrent mode for inference — where both produce identical outputs. The recurrent mode enables constant-time, constant-memory generation per token. xLSTM has been applied to time series forecasting, image segmentation, and audio representation learning, but to the best of our knowledge, its application to symbolic music generation has not been previously explored.

## Methodology

Our approach treats symbolic music generation as a next-token prediction task.

**REMIGEN Encoding**: MIDI files are converted into flat sequences of discrete tokens using the REMIGEN encoding. Each token uses a prefix-value format: `s` for time signature, `t` for tempo (log-quantized BPM), `o` for bar offset, `i` for instrument, `p` for pitch, `d` for duration, and `v` for velocity. Each note is represented as a pitch-duration-velocity (p-d-v) triplet, forming a strict grammar that the model must learn to follow.

**Dataset**: We use the Lakh MIDI Dataset (LMD), encoding 45,043 MIDI files into approximately 611.6 million tokens, split into training (36,034 songs), validation (4,504 songs), and test (4,505 songs) sets.

**Model Architecture**: We train a 12-block xLSTM model with 512-dimensional embeddings and a context length of 4,096 tokens using the Helibrunna training framework. The architecture uses a mix of mLSTM and sLSTM blocks, with sLSTM placed at layers 3, 6, and 9.

**Recurrent Inference**: We identified that Helibrunna's generation loop uses xLSTM's parallel formulation — reprocessing the entire sequence at each step — resulting in quadratic time complexity and out-of-memory errors. We implemented a custom recurrent-state generator that uses the model's native recurrent inference mode, reducing complexity from O(N²) to O(N) with constant memory usage. The recurrent generator was verified to produce identical outputs to the parallel generator through a controlled equivalence test.

**Grammar-Aware Token Filtering**: During generation, the model occasionally produces tokens that violate the REMIGEN p-d-v triplet grammar. We apply a post-generation repair step that removes incomplete or orphaned tokens, ensuring all outputs can be decoded to MIDI.

## Experiment Setup and Implementation

We evaluated the system along several dimensions:

**Perplexity and Extrapolation**: We measured perplexity on the validation set at context lengths from 1,024 to 10,240 tokens to assess the model's ability to extrapolate beyond its training context of 4,096 tokens.

**Generation Process Evaluation**: We generated 30 pieces at each of four target lengths (2,048, 4,096, 8,192, and 12,288 tokens) using checkpoint 46,000 with temperature 0.9, and evaluated:
- Musical output properties: number of bars, number of notes, and token density per bar
- Generation performance: wall-clock generation time and throughput (tokens per second)
- Token grammar quality: percentage of tokens violating the REMIGEN grammar before filtering

**Inference Benchmark**: We compared generation times between the original parallel-mode generator and our recurrent-state generator across sequence lengths from 100 to 12,000 tokens.

<!-- Chamodi: Add details about musical/structural quality evaluation setup -->

## Results and Analysis

**Perplexity and Extrapolation**: The model achieves its lowest perplexity of 1.65 at a context length of 5,120 tokens — beyond the training context of 4,096 — demonstrating meaningful extrapolation. Perplexity gradually increases to 1.89 at 10,240 tokens. Compared to Transformer-based models reported by Museformer, our xLSTM achieves competitive perplexity at shorter contexts (1.80 at 1,024) but does not match Museformer's strong results at longer contexts, where its structure-aware attention provides a clear advantage.

**Musical Output Properties**: Both the number of bars and notes scale approximately linearly with target token length, indicating the model consistently produces musical content rather than degenerating at longer lengths. Token density per bar remains stable with a mean of 128.6 tokens per bar.

**Generation Performance**: Generation time scales linearly with sequence length, and throughput remains constant at a mean of 144.2 tokens per second — a direct consequence of the recurrent formulation's O(1) cost per token.

**Token Grammar Quality**: Grammar error rates are effectively zero within the training context length, increasing to approximately 2.3% at 8,192 tokens and 13% at 12,288 tokens. The large standard deviation at 12,288 tokens indicates high variability across pieces. The grammar-aware filtering stage ensures all outputs can be decoded.

**Inference Speedup**: The recurrent generator exhibits linear scaling while the parallel approach shows quadratic scaling. Speedups grow with sequence length, and the recurrent generator eliminates the OOM errors that previously prevented long-form generation.

<!-- Chamodi: Add musical/structural quality results -->

## Conclusion

We demonstrated that xLSTM is a viable architecture for symbolic music generation. The model can be trained on large-scale MIDI data and generates musically proportionate output across a range of sequence lengths. By implementing recurrent-state inference, we reduced generation complexity from O(N²) to O(N), enabling the generation of long pieces that were previously infeasible. The model shows meaningful extrapolation beyond its training context, though grammar adherence degrades gradually at longer sequences. Our findings suggest that modern recurrent architectures offer a promising and complementary direction to Transformer-based approaches for long-form music generation.

Future directions include training with longer context lengths, exploring different xLSTM configurations, investigating constrained decoding strategies, and conducting more extensive comparisons with Transformer-based models on identical datasets.

## Publications

[//]: # "Note: Uncomment each once you uploaded the files to the repository"

<!-- 1. [Semester 7 report](./) -->
<!-- 2. [Semester 7 slides](./) -->
<!-- 3. [Semester 8 report](./) -->
<!-- 4. [Semester 8 slides](./) -->
<!-- 5. Haritha Bandara, Yohan Senanayake and Chamodi Senaratne, "Exploring xLSTM for Long-Term Structural Coherence in Symbolic Music Generation" (2026). [PDF](./). -->

## Links

[//]: # ( NOTE: EDIT THIS LINKS WITH YOUR REPO DETAILS )

- [Project Repository](https://github.com/cepdnaclk/e20-4yp-symbolic-music-generation-with-xLSTM/tree/main)
- [Project Page](https://cepdnaclk.github.io/e20-4yp-Exploring-xLSTM-for-Long-Term-Structural-Coherence-in-Symbolic-Music-Generation)
- [Department of Computer Engineering](http://www.ce.pdn.ac.lk/)
- [University of Peradeniya](https://eng.pdn.ac.lk/)
