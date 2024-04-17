#!/usr/bin/env python

import os
import os.path as osp
import sys
import subprocess


def exec_cmd(command: str) -> None:
    assert len(command) > 0, 'Command is empty'
    process = subprocess.Popen(
        command,
        stdout=sys.stdout,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    process.communicate()
    assert process.returncode == 0, f'Command {command} failed with returncode {process.returncode}!'


def install(current_directory):
    directory = osp.join(current_directory, 'dab_detr/models/dab_deformable_detr/ops/')
    assert osp.isdir(directory), f'{directory} not found'

    os.chdir(directory)

    bash_command = './make.sh'
    assert osp.isfile(bash_command)
    exec_cmd(bash_command)

    
try:
    current_directory = osp.dirname(osp.abspath(__file__))
    import MultiScaleDeformableAttention  # NOQA: F401
except ImportError:

    install(current_directory)


sys.path.append(osp.join(current_directory, 'dab_detr'))
