# CHASTE: Disclaimer

CHASTE (Constrained Holistic Annotator System with onTological Entities) is a research pipeline that extracts entities and relations from biomedical abstracts with grammar-constrained language models; it has no clinical function.

> **Research use only. Not a medical device.**
> This software was developed in the HEREDITARY project as a research prototype. It has not been assessed or certified under the EU Medical Device Regulation (EU) 2017/745. It must not be used to diagnose, treat, monitor or make any other decision about individual patients.

## 1. Overview

| Field | Value |
|---|---|
| Name | CHASTE (Constrained Holistic Annotator System with onTological Entities); the repository also holds the 2025 entry CLEANR |
| Version | submission-licensed (git tag, archived on Zenodo). The beam-search sub-package is tagged separately (beams-v0.1.2, PyPI `llama-cpp-beam-search` 0.1.2). |
| Type | Python research code and notebooks: information-extraction pipeline with locally run language models; no user interface |
| Owner partner and contact | Institute of Visual Computing (IVC), TU Graz; Benedikt Kantz, benedikt.kantz@tugraz.at |
| Licence | MIT, see [LICENSE](LICENSE). Base models and data carry their own terms, see Sections 3 and 4. |
| Repository / DOI / demo | Repository: https://github.com/Dakantz/CHASTE · Paper: CLEF 2026 Working Notes (CEUR-WS), no DOI; preprint https://dakantz.at/papers/2026/chaste/paper.pdf · Zenodo: https://doi.org/10.5281/zenodo.20313871 (submission-licensed) · Demo: none |
| Release date | 2026-05-20 (tag submission-licensed) |
| Related HEREDITARY task | Not confirmed: the project publication record lists the paper under WP4; D5.5 names Task 5.6 for the GutBrainIE dataset |
| How to cite | Kantz, Waldert, Lengauer, Schreck, *CHASTE: Constrained Holistic Annotator System with onTological Entities*, CLEF 2026 Working Notes; software: doi 10.5281/zenodo.20313871. See [citation.cff](citation.cff) and the README. |

## 2. Intended use

**Intended use.** Research on information extraction from biomedical literature: running and comparing grammar-constrained, beam-search and lookup-based pipelines for named entity recognition, entity linking and relation extraction on the GutBrainIE benchmark, and reproducing the results of the paper.

**Intended users.** Researchers in biomedical natural language processing and knowledge-graph construction; developers building on this code.

**Out-of-scope uses.** This software must not be used for
- clinical decisions, i.e. diagnosis, prognosis, treatment or monitoring of individual patients, or as a source of medical advice;
- decisions about individuals, such as insurance, employment or access to care;
- re-identification of individuals or linkage with individual-level data;
- other data types than published scientific abstracts, e.g. clinical notes, health records or other text about individual patients;
- presenting extracted relations as established biomedical facts without checking them against the source articles.

## 3. Data provenance

The repository contains no patient data and none of the GutBrainIE dataset. Earlier commits in the git history did contain GutBrainIE files, see Section 7.

| Dataset | Source | Type | Population | Licence / access | In the repo? |
|---|---|---|---|---|---|
| GutBrainIE 2026 (train: gold, silver, silver 2025, bronze; dev; test) | GutBrainIE task organisers (University of Padua, HEREDITARY), BioASQ Lab at CLEF 2026 | Public PubMed titles and abstracts with bibliographic metadata, and expert, student and distantly supervised annotations | Published articles on the gut-brain axis; no individual-level data | Available after registration to the BioASQ Lab; no redistribution licence found | No, place it under `data/` yourself |
| `data/bionner/dev_relations.tsv` | Derived from the NEREL-BIO development set (https://github.com/nerel-ds/NEREL-BIO) | Public annotated biomedical abstracts | Not applicable | Not established; the NEREL-BIO repository states no licence | Yes |
| `submission/results_test/*.csv` | GutBrainIE official result tables for our runs and the merged runs | Evaluation scores | Not applicable | Not applicable | Yes |

No ethics approval is needed for the data above, as none of it is individual-level data held by the project.

## 4. Method

- **What is computed.** Sentences of each title and abstract are tagged with candidate entity spans (spaCy `en_core_web_hftrf`, or a BERT token tagger fine-tuned from `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` in `finetune_ned.ipynb`). A GBNF grammar per sentence restricts a Hermes 3 model to these spans and the task's labels and predicates; generation runs through `llama-cpp-python`, optionally with a constrained beam search and with retrieved training sentences (embeddings from `NeuML/pubmedbert-base-embeddings`). Entities are linked to concept URIs by exact and TF-IDF matching against the training annotations. A lookup baseline matches training-set entities verbatim.
- **Training.** Optional LoRA fine-tuning of NousResearch Hermes 3 (Llama 3.2 3B and Llama 3.1 8B) on the GutBrainIE training collections with torchtune (`configs/`), and fine-tuning of the BERT span tagger.
- **Released weights.** None. The LoRA adapters, the quantised GGUF models and the fine-tuned tagger stay local (`finetunes/`, `quants/`, `finetuned/` are not committed and not uploaded to Hugging Face). Anyone who re-creates them is bound by the base model terms: Hermes 3 is a Llama derivative under the Llama 3.1 / Llama 3.2 Community License and Acceptable Use Policy (the `meta-llama` tokenizers are gated on Hugging Face); BiomedBERT is MIT; `NeuML/pubmedbert-base-embeddings` is Apache-2.0.
- **Privacy techniques.** None; no personal data is processed.
- **Outside services.** All inference and training run locally (workstation or Slurm cluster). The code contacts only:
  - the Hugging Face Hub, to download base models, tokenizers, the embedding model and, with `Llama.from_pretrained`, GGUF files (with your Hugging Face token for the gated `meta-llama` repositories);
  - at install time, GitHub (submodules, the `llama-cpp-python` fork, the spaCy model wheel) and the Python package indexes.
  - No LLM API is called. `openai` and `langchain[openai]` are still declared dependencies and `slurm/evaluate_openai.sh` still exists, but the OpenAI code path in `inference.py` is commented out.

## 5. Performance

All numbers are from the paper (micro-averaged F1, the task's ranking metric; GutBrainIE 2026, 80 development and 80 test abstracts).

| Subtask | Best own test run | Organisers' baseline (test) | Best own dev run |
|---|---|---|---|
| 6.1.1 NER | 0.44 | 0.80 | 0.63 |
| 6.1.2 NERD (with URIs) | 0.31 | 0.44 | 0.56 |
| 6.2.1 RE | 0.11 | 0.39 | 0.10 |
| 6.2.2 Mention-level RE | 0.06 | 0.13 | 0.09 |

- The fine-tuned PubMedBERT span tagger reaches F1 0.72 on the dev set, against 0.48 for spaCy; its recall caps the whole pipeline.
- No CHASTE run beats the organisers' baseline. Results drop markedly from dev to test. Beam search raised precision and lowered recall on dev and could not be submitted for the test set because of its run time.
- Intersecting our runs with those of the Graphwise team gave the most precise NERD submission on the test set (micro precision 0.77, F1 0.56).

## 6. Known limitations and risks

- **Generalisability.** Developed and evaluated only on English PubMed abstracts about the gut-brain axis and the GutBrainIE label set; dev and test sets have 80 abstracts each. Performance on other text types or domains is unknown.
- **Biases.** The training annotations mix expert, student and automatically generated (bronze) labels; errors and gaps in them are learned. The literature over-represents well-studied diseases and model organisms.
- **Failure modes.** Entities the span tagger misses can never be extracted. The model can assign wrong labels, link to wrong URIs or produce relations that the abstract does not state; relation extraction F1 is about 0.1. An extracted relation states only that a sentence was read that way, not that the relation holds.
- **Privacy.** No personal data is processed and no model is trained on personal data, so no privacy guarantee is claimed or needed. The dataset's article metadata contains author names of the publications.

## 7. Responsible use

- **Terms of use.** The code is MIT licensed. Use the GutBrainIE data only under the organisers' conditions, and the base models under their licences (Llama 3.1 / 3.2 Community License and Acceptable Use Policy for Hermes 3). Check extracted relations against the source abstracts before reusing them.
- **Open items before release.** The git history (commits from 2025-02 to 2026-05, contained in all tags including the Zenodo archive) still holds GutBrainIE 2025 training and development files, the 2026 raw test articles and the 2026 annotation guidelines; whether this redistribution is permitted must be clarified with the organisers. The licence of the NEREL-BIO-derived `data/bionner/dev_relations.tsv` is not established. The Zenodo archive (tag submission-licensed) contains a LICENSE file without a copyright line. Slurm scripts name cluster user accounts.
- **Contact for misuse or vulnerability reports.** Benedikt Kantz, benedikt.kantz@tugraz.at

## 8. Acknowledgement

Funded by the European Union under grant agreement No 101137074 ([HEREDITARY](https://hereditary-project.eu/)).
Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Health and Digital Executive Agency (HaDEA).
Neither the European Union nor the granting authority can be held responsible for them.
