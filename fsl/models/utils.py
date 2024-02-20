#!/usr/bin/env python

import os
import subprocess


def download(url: str, directory: str = './') -> str:
    os.makedirs(directory, exist_ok=True)
    try:
        command = ['wget', url, '-P', directory, '--quiet', '--show-progress', '--progress=dot']
        subprocess.run(command, check=True)
        print(f'Downloaded {url}')
    except subprocess.CalledProcessError as e:
        print(f'Error downloading {url}: {e}')
    return directory
