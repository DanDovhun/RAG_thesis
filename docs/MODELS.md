# DENSE EMBEDDING MODELS
## Local Dense Embedding Models:

* SlovakBERT
  * [gerulata/slovakbert](https://huggingface.co/gerulata/slovakbert)
* czech-bert-base-cased
  * [czech-original-fold2-bert-base-cased](https://huggingface.co/spneshaei/czech-original-fold2-bert-base-cased)
* xlm-roberta-base
    * [FacebookAI/twitter-xlm-roberta-base](https://huggingface.co/FacebookAI/twitter-xlm-roberta-base)
* bert-base-multilingual-cased
    * [google-bert/bert-base-multilingual-cased](https://huggingface.co/google-bert/bert-base-multilingual-cased)
* Qwen3-Embedding 6b
    * [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)

### Installation of local models:
```sh
cd /path/to/RAG_thesis      # Only outside RAG_thesis
source .venv/bin/activate
python3 download_models.py
```

## Cloud Based Dense Embedding Models:
* text-embedding-3-small
    * Owner: OpenAI
* test-embedding-3-large
    * Owneer: OpenAI

## List of all embedders:
List of all embedders used for the research can be found in [embedder_config](../embedder_config.toml)