from pathlib import Path
from spikeinterface.sorters import get_default_sorter_params

DATA_ROOT = Path("F:/Data_Mice_IBL")
DB_PATH = Path(r"F:\Data_Mice_IBL\full_db_all_rigs.feather")

KS_PARAMS = get_default_sorter_params("kilosort4")

COMPRESS_KEEP_ORIGINAL = False
KS_REMOVE_EXISTING_FOLDER = True

ANALYZER_N_JOBS = 1
ANALYZER_CHUNK_DURATION = "1s"

EXPORT_N_JOBS = 1
EXPORT_CHUNK_DURATION = "1s"