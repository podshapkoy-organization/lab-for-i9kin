import torch

TRAIN_FOLDER = "train/"
TEST_FOLDER = "test/"
TARGET_FILE = "train/targets.tsv"
TEST_TARGET_FILE = "test/targets.tsv"

MFCC_N = 15
CACHE_TRAIN = "train_cache.pkl"
CACHE_TEST = "test_cache.pkl"

EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LSTM_HIDDEN_SIZE = 128
NUM_CLASSES = 2

LOG_DIR = "logs"
