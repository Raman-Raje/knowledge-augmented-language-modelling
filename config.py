
import warnings

warnings.filterwarnings("ignore")

# Data path
DATA_PATH = "./CONLL2003/"

# Standard parameters
TRAIN_DP = 16
TEST_DP = 8
BATCH_SIZE = 16
embedding_dim = 400
dec_units = 1150
WH_UNITS = WE_UNITS = 100
VOCAB_SIZE = 5000
NB_ENTITIES = 4
EPOCHS = 1