from os import environ

home_folder = environ.get('HOME')
DATASET_FOLDER = f"{home_folder}/skdi_dataset"
TRAINING_FOLDER = f'{DATASET_FOLDER}/Training'
TESTING_FOLDER = f'{DATASET_FOLDER}/Testing'

