import glob
import os
import shutil
from pathlib import Path

import nam.cli
import json
from nam.train.core import _calibrate_latency_v3, _check_v3
from nam.data import wav_to_np
from nam.train.full import main as nam_full
from namscripts.resources import get_resource

CONFIGS = {
    'default': get_resource('configs/nam_default.json')
}

MAX_EPOCHS = 100


def main():
    base_dir = 'D:\\CaptureTraining\\gp1000\\8_19_20_scr'

    out_dir = Path(base_dir, 'out')
    input_dir = Path(base_dir, 'src')
    make_dirs(out_dir)
    cfg = setup(base_dir, get_resource("v3_0_0.wav"), out_dir)

    for file in glob.glob(os.path.join(input_dir, '*.wav')):
        run_single(cfg, file)
        print("|||  ")
        # break


def calibrate(output_path):
    calibration_output = _calibrate_latency_v3(wav_to_np(str(output_path)))
    # print(f"Calibrated and setting delay to: {calibration_output.recommended}")
    return calibration_output.recommended


def run_single(config, file):

    delay = calibrate(file)
    _check_v3(config['data']['common']['x_path'], file, True)

    config['data']['common']['y_path'] = file
    config['data']['common']['delay'] = delay

    nam_full(
        config['data'],
        config['model'],
        config['learning'],
        Path(config['out_dir']),
        False
    )

def setup(base_dir, input_path, out_dir):
    log_script = get_resource("tensorlogs.bat")
    try:
        shutil.copy(log_script, out_dir)
    except Exception as e:
        print(e)

    config = CONFIGS['default']
    with open(config, 'r') as f:
        cfg = json.load(f)
        cfg['data']['common']['x_path'] = input_path
        cfg['data']['common']['y_path'] = None
        cfg['learning']['trainer']['max_epochs'] = MAX_EPOCHS
        cfg['base_dir'] = base_dir
        cfg['out_dir'] = str(out_dir)

        # config_export = os.path.join(base_dir, 'config.json')
        with open(Path(base_dir, 'full_config.json'), 'w') as cf:
            json.dump(cfg, cf, indent=4)

        return cfg


def make_dirs(dir):
    try:
        os.makedirs(dir, exist_ok=True)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
