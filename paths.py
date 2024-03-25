from os import environ

home_folder = environ.get('HOME')
WORKSPACE = f'{home_folder}/Skin_Disease_Detection'
DATASET_FOLDER = f"{home_folder}/skdi_dataset"
TRAINING_FOLDER = f'{DATASET_FOLDER}/Training'
TESTING_FOLDER = f'{DATASET_FOLDER}/Testing'
CHECKPOINT_FOLDER = f'{WORKSPACE}/checkpoints'
STATUS_FOLDER = f'{WORKSPACE}/status'
PLOT_FOLDER = f'{WORKSPACE}/plots'
PARAM_FOLDER = f'{WORKSPACE}/param'
CONFIG_FOLDER = f'{WORKSPACE}/config'

