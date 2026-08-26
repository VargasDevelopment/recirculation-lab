"""Pinned inputs and the paper's fixed Gemma 3 1B configuration."""

MODEL_ID = "google/gemma-3-1b-pt"
MODEL_REVISION = "fcf18a2a879aab110ca39f8bffbccd5d49d8eb29"

# Small parquet mirror of the original PG-19 test split. The paper does not
# identify its data artifact revision, so this one is deliberately explicit.
DATASET_ID = "emozilla/pg19-test"
DATASET_REVISION = "c5e39bf32e33f9111323aa68d7d9000d22722035"
DATASET_SPLIT = "test"

CONTEXT_LENGTH = 1024
NUM_DOCUMENTS = 2
WINDOWS_PER_DOCUMENT = 5

DESTINATION_LAYER = 4
SOURCE_LAYER = 11
ALPHA = 0.15
BETA = 0.85
NUM_RECURRENCE_STEPS = 1
NORMALIZATION = "norm_rep"
RAMP_STEPS = 10

DTYPE_NAME = "bfloat16"
DEVICE_NAME = "mps"
ATTENTION_IMPLEMENTATION = "eager"
