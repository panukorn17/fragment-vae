import json
import pandas as pd
import pickle as pkl
import subprocess


def load_pickle(path):
    return pkl.load(open(path, "rb"))


def save_pickle(obj, path):
    pkl.dump(obj, open(path, "wb"))


def load_json(path):
    return json.load(open(path, "r"))


def save_json(obj, path):
    json.dump(obj, open(path, "w"), indent=2)


def commit(experiment_name, time):
    """
    Try to commit repo exactly as it is when starting
    the experiment for reproducibility.
    """
    try:
        # Construct the commit message
        commit_message = f"auto commit tracked files for new experiment: {experiment_name} on {time}"

        # Run the git commit command
        subprocess.run(['git', 'commit', '-a', '-m', commit_message, '--allow-empty'], check=True)

        # Get the current commit hash
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()

        return commit_hash
    except Exception as e:
        print(f"Error: {e}")
        return '<Unable to commit>'


def load_dataset(config, kind):
    assert kind in ['train', 'test']
    path = config.path('data')
    filename = path / f'{kind}.smi'
    return pd.read_csv(filename, index_col=0)
