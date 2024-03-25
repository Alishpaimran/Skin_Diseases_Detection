from os import environ

home_folder = environ.get('HOME')
WORKSPACE = f'{home_folder}/Skin_Disease_Detection'
DATASET_FOLDER = f"{home_folder}/skdi_dataset"
TRAINING_FOLDER = f'{DATASET_FOLDER}/Training'
TESTING_FOLDER = f'{DATASET_FOLDER}/Testing'
CHECKPOINT_FOLDER = f'{WORKSPACE}/checkpoints'
STATUS_FILE = f'{WORKSPACE}/status.txt'

