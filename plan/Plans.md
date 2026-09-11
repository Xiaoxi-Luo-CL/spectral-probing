# Plans

## Motivation

Language can be viewed as a multi-scale sequence signal, where different types of information—such as lexical, syntactic, and semantic—are distributed across different scales. Prior work and our preliminary results suggest that this structure can be partially revealed through frequency analysis of model representations.

Based on this, we ask: (1) is there a shared patterns in how information is distributed across scales across languages? Are these patterns more similar within relative languages? (2) Are language-specific deviations that reflect underlying linguistic properties? If such structure exists, the shared patterns may facilitate cross-lingual alignment, while the language-specific patterns can provide insights into how LLMs encode linguistic information and facilitate interpretability.

[real-life influence: align languages better/with less data, interpretability]

---

I would like clarify an important distinction between the principle of processing ”spectrum of the original embedding” and “spectral profile”. For the former, we hope the embedding of parallel sentences to be as close as possible, “the degree of alignment is causally predictive of downstream cross-lingual transfer” (Cao 2020). For the latter, “spectral profiles” (weights of different frequencies) are always dependent on specific tasks, reflecting which frequencies carry useful information for that task. It is natural for different languages to have different spectral profiles, and we should not force them to match across languages. 

The proposed directions are organized roughly in order of **increasing methodological complexity**.

## Direction 1: Spectral Profile for Zero-shot Cross-lingual Transfer

### Motivation

We consider the standard cross-lingual transfer setting: given a high-resource language A and task T, we aim to transfer to a low-resource language B on the same task, where only limited annotated data in B is available. We ask:

> Does the spectral profile learned from (A, T) facilitate transfer to (B, T)? More specifically, does spectral similarity between languages predict transfer performance?
> 

This leads to a central question: Do spectral profiles primarily capture **task-specific information**, **language-specific information**, or a combination of both?

### **Existing results**

After word-level pooling (to control for fertility), I observe that in encoder-only models, spectral profiles for parsing head relations are highly similar across languages, while in decoder-only models, spectral profiles can differ substantially across languages. This suggests:

- In encoder-only models, spectral profiles may reflect a **language-invariant characterization of the task**, potentially supporting transfer.
- In decoder-only models, spectral profiles may encode more **language-specific properties** (or training artifacts that we didn’t explore yet), which may hinder transfer.

### Experimental Design

1. For task T and language A, train a probe on pre-trained multilingual models, get spectral profile P_A. Record $acc_A$. Compare:
    1.  Evaluate [$P_A$  + the probe trained on A] on B
    2. Apply the probe trained on A directly to B, without $P_A$
    3. Train a probe using the limited annotated data from B
2. [for encoder-only models] Treat the spectral profile (after word-level pooling) as a language-invariant representation for task T. Conduct the same word-level pooling in language B and apply the language-irrelevant spectral profile on word-level representations of B.
3. Is (a) *distance in spectral profile*, or (b) *typological similarity* a good predictor for cross-lingual transfer ability?

examples: Mandarin to Yue and Min transfer, Turkish to Turkmen, Hindi to other Indian languages (e.g., Malayalam, Marathi), 

## Direction 2: Frequency-Weighted Cross-Lingual Representation Alignment

### Motivation

A common approach to improving cross-lingual transfer is to explicitly align representations of parallel sentences across languages (e.g., Cao et al., 2020; Shen et al., 2025). This is typically implemented by adding alignment regularization **in (continue) pre-training**.

I propose to extend this approach by performing alignment in the **frequency domain**, motivated by the hypothesis that different frequency components capture information at different scales and may contribute unequally to cross-lingual transfer.

### Method

Given parallel sentences in source language s and target language t, we first obtain their representations h_s and h_t.

 $F_s=\text{DCT}(h_s), F_t=\text{DCT}(h_t)$

We then define a frequency-weighted alignment objective:

$$
L_{align} = \sum_{s, t}\sum_k w_k · \frac{|F_s[k] − F_t[k]|}{|F_s[k] + F_t[k] + \epsilon|}
$$

where F[k] is the k-th DCT component of the representation sequence, and $w_k$ is the importance of frequency k for alignment. We need to constraint relative frequency wight because the scales of different frequencies’ coefficients after DCT vary greatly.

In addition to this pointwise alignment loss, I will also explore **contrastive objectives in frequency space**, which is more commonly used in representation alignment (e.g., Liu and Niehues, 2025). 

Aligning middle-layer representations is reported to achieve more balanced improvements in transfer languages on Qwen and Llama (Liu and Niehues, 2025). I will also consider aligning different layers.

An even stricter version would be aligning the corresponding frequency components (after inverse DCT). *(Note: I think it is closest to our original cross-modal alignment idea.)*

$$
L_{align} = \sum_{s, t}\sum_k w_k · \|\text{IDCT}[F_s[k]] − \text{IDCT}[F_t[k]]\|
$$

#### Hypothesis 1: Spectral alignment improves transfer

This is the most trivial case, $w_k = 1$ for all k.

#### Hypothesis 2: Low-frequency bands are more important to alignment

I will test this by: 1) letting $w_k$ inversely proportional to frequency 2) performing ablations over subsets of frequency bands.

#### Hypothesis 3: Align frequency bands that are most informative for the downstream task

If the source language is high-resourced, we could obtain its spectral profile of the downstream task. Then, we could set 
$w_k$ based on its spectral profile of the target task at frequency k. 

#### Things to notice

Though we do not plan to include generation tasks at present, it is worth noticing that overly aggressive alignment harms generation tasks.

> “… the reason why zero-shot cross-lingual transfer in text generation fails is because the fine-tuning process learns language invariant representations, which is beneficial for classification tasks, but detrimental to generation tasks” (Li and Murray, 2023).
> 

## Direction 3: Frequency-incorporated Modular Adapters

![image.png](image.png)

### Motivation

Modular adaptation frameworks (e.g., MAD-X and its variants) decompose cross-lingual transfer into **language-specific** and **task-specific** components. While recent PEFT methods (e.g., LoRA) are widely used, they typically do not explicitly separate language and task factors. Therefore, I think MAD-X (Pfeiffer et al., 2020) is a valuable starting point, as separating language from task adaptation makes cross-lingual transfer reusable.

### Method

I propose to incorporate **frequency-structured transformations** into the language adaptation component. Given hidden representation $h \in \mathbb{R}^{s\times d}$, where s is sequence length, d is feature dimension:

1. Transform to frequency space: $H_f=\mathrm{DCT}(h)$
2. Apply a learnable frequency **gate:** $H_f' = H_f + g_{\ell} \odot H_f$, where $g_{\ell} \in \mathbb{R}^s$ is a language-specific parameter over frequencies.
3. Transform back: $h' = \mathrm{IDCT}(H_f')$
4. Integrate with MAD-X modules:

 $h'=h+\rm{FreqAdapter}(h)+\text{Lang Adapter}(h)+\text{Task Adapter}(h)$

#### Hypothesis 1: Frequency adapter improves zero-shot cross-lingual transfer.

We follow a modular training pipeline:

1. **Language adapter:** Train language adapter (including frequency adapters) on **unlabeled data** for both source and target language, get FA_s, FA_t, LA_s and LA_t. (MAD-X is based on encoder-only model, so its training task for this step is masked language modeling. We can simply use next-token prediction here)
2. **Task adapter:** Train task adapter on labeled data from the source language, get TA.
3. **Zero-shot transfer:** Combine the FA_t, LA_t + TA for evaluation.

Compare: (1) full fine-tuning, (2) standard adapter baseline (e.g., MAD-X) (3) the frequency adapter alone, (4) frequency adapter + MAD-X, and (5) PEFT methods, e.g., LoRA.

#### Hypothesis 2:  $g_{\ell} \in \mathbb{R}^s$ is be similar between topologically similar languages

Languages with similar statistical or structural properties are expected to learn similar frequency gates. This structure can help **low-resource transfer**.

**Experiment**: Use the gate from a related language as a proxy for a low-resource target language, or average gates from multiple related languages

A recent work ([https://arxiv.org/pdf/2601.16629](https://arxiv.org/pdf/2601.16629)) shows that typological similarity can be used to aggregate language adapters and create useful proxy modules for unseen languages, but it focuses on encoder-only models. Given the observed differences between encoder-only and decoder-only models in spectral behavior, extending this analysis is still meaningful.

## Direction 4: Relate spectral profiles of decoder-based models to typology

As discussed in Direction 1, I restrict the scope of investigation to “decoder-based models” because they may encode more language-specific information (or artifacts) in spectral profiles. However, there could be much noise (e.g., tokenization) in spectral profiles, making fine-grained linguistic analysis difficult. 

There are still some possible ideas, though I would not prioritize at present:

1. Are linguistics features (e.g., word order features from WALS) reflected by spectral profiles in corresponding tasks (need some design but could be syntax-related)?
2. Which bands are most predictive of each typological feature? 

Typology analysis should control for at least fertility and maybe script class, making it more difficult.

### Reference

Cao et al., 2020, Multilingual Alignment of Contextual Word Representations. In ICLR

Pfeiffer et al., 2020, MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer. In EMNLP.

Liu and Niehues, 2025, Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs. In ACL [https://aclanthology.org/2025.acl-long.778.pdf](https://aclanthology.org/2025.acl-long.778.pdf)

Shen et al., 2025, From Unaligned to Aligned: Scaling Multilingual LLMs with Multi-Way
Parallel Corpora, in EMNLP. [https://aclanthology.org/2025.emnlp-main.374.pdf](https://aclanthology.org/2025.emnlp-main.374.pdf)

Li and Murray, 2023, Why Does Zero-Shot Cross-Lingual Generation Fail? An Explanation and a Solution, in ACL. [https://aclanthology.org/2023.findings-acl.789.pdf](https://aclanthology.org/2023.findings-acl.789.pdf)