"""Pinned inputs for the Gemma 3 1B IT capability-transfer experiment."""

IT_MODEL_ID = "google/gemma-3-1b-it"
IT_MODEL_REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
IT_TOKENIZER_REVISION = IT_MODEL_REVISION
IT_WEIGHTS_SHA256 = "3d4ef8d71c14db7e448a09ebe891cfb6bf32c57a9b44499ae0d1c098e48516b6"
IT_TOKENIZER_JSON_SHA256 = (
    "4667f2089529e8e7657cfb6d1c19910ae71ff5f28aa7ab2ff2763330affad795"
)
IT_CHAT_TEMPLATE_SHA256 = (
    "7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4"
)

HARNESS_REPOSITORY = "https://github.com/EleutherAI/lm-evaluation-harness"
HARNESS_REVISION = "dd417662e5bda6a247489ec28d2ff46a45d1c42c"

BENCHMARKS = {
    "mmlu_pro": {
        "dataset_id": "TIGER-Lab/MMLU-Pro",
        "dataset_revision": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
        "task_version": "3.1",
    },
    "gsm8k": {
        "dataset_id": "openai/gsm8k",
        "dataset_revision": "740312add88f781978c0658806c59bc2815b9866",
        "task_version": "3.0",
    },
    "ifeval": {
        "dataset_id": "google/IFEval",
        "dataset_revision": "966cd89545d6b6acfd7638bc708b98261ca58e84",
        "task_version": "4.0",
    },
    "hellaswag": {
        "dataset_id": "Rowan/hellaswag",
        "dataset_revision": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
        "task_version": "1.0",
    },
}

CAPABILITY_SEED = 20260826
CAPABILITY_DTYPE = "bfloat16"
CAPABILITY_DEVICE = "mps"
CAPABILITY_ATTENTION_IMPLEMENTATION = "eager"
CAPABILITY_MAX_LENGTH = 32768
