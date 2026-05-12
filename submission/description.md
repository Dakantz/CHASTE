# GutBrain IE Challenge @ CLEF 2025: {{system_id}}

* Team ID: {{team_id}}
* TaskID: {{task_id}}
* RunID: {{run_id}}
* Run Flags
{{#flags}}
  - {{.}}
{{/flags}}
* GitHub: https://github.com/Dakantz/CHASTE
## Our appraoch
* Use a RAG approach to prompt a LM to return the relations
  - fetch similar articles from VectorDB to give good examples (if the run ID contains `rag`)
  - finetune the Hermes model on the train data combinations, with text+annotation pairs (if the run ID contains `lora`)
  - use an efficient grammar-based generation strategy for both entities and relations
  - the grammar is informed from NE from `spacy` or a finetuned model if the run id contain `finetuneed`
  - disambiguation uses simple TF-IDF matching
  - we also employ beam search over tokens, either:
    - branching over 2 paths for 3 depth steps and taking the maximal likelihood, or
    - generating the full relation/entity for 4 branches on the inital step, and 2 more for the next one, then generate until the end and evaluating the complete likelihood

* we also performa a `naive` baseline matching just the train dataset entities directly to the sentence entities, achieving good results at almost no compute cost