import dnnutil
import os
import shutil
from pathlib import Path
import json


def make_template(args):
    '''TODO:docs
    '''
    if os.path.isfile(args.file):
        while True:
            resp = input(
                '{} already exists. Overwrite it [y|N]? '.format(args.file))
            if resp.lower() in ['y', 'n']:
                break
            elif resp == '':
                resp = 'n'
        if resp == 'n':
            return

    with open(dnnutil.template, 'r') as f:
        temp = f.read()
    with open(args.file, 'w') as f:
        f.write(temp)


def make_config(args):
    '''TODO:docs
    '''
    conf = Path(args.file).with_suffix('.json')
    if conf.is_file():
        while True:
            resp = input(
                '{} already exists. Overwrite it [y|N]? '.format(str(conf)))
            if resp.lower() in ['y', 'n']:
                break
            elif resp == '':
                resp = 'n'
        if resp == 'n':
            return

    with open(dnnutil.config, 'r') as f:
        temp = f.read()
    with conf.open('w') as f:
        f.write(temp)


def purge(args):
    '''TODO:docs
    '''
    rundir = Path(args.dir)
    for run in rundir.iterdir():
        if run.is_dir() and run.stem.isnumeric():
            if (not run.joinpath('.state').is_file() and
                not run.joinpath('model_weights').is_file()):
                for f in run.iterdir():
                    os.remove(f)
                os.removedirs(run)
                print(f'Delete {run.name}')
    runs = sorted([x for x in rundir.iterdir() if x.stem.isnumeric()],
                  key=lambda x:int(x.name))
    for i, run in enumerate(runs):
        if int(run.name) != i + 1:
            shutil.move(run, run.with_name(str(i + 1)))
            print(f'Move {run.name} -> {i + 1}')


def desc(args):
    '''TODO:docs
    '''
    rundir = Path(args.dir)
    runs = sorted([x for x in rundir.iterdir() if x.stem.isnumeric()],
                  key=lambda x:int(x.name))
    for run in runs:
        try:
            j = json.load(open(run / 'config.json'))
            desc, deets = dnnutil.config_string(j)
            print(f'Run id {run.name:>2}:  {desc}')
            if args.verbose:
                print(deets)
        except Exception as e:
            print(f'Run id {run.name:>2}:  (unable to read a config file)')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers()

    parse_new = subs.add_parser('new', aliases=['n'],
        help='Generate a template main file.')
    parse_new.add_argument('-f', '--file', type=str, default='main.py',
        help='Name (location) of output file. Default: ./main.py')
    parse_new.set_defaults(func=make_template)

    parse_conf = subs.add_parser('conf', aliases=['c'],
        help='Generate a template configuration file.')
    parse_conf.add_argument('file', type=str,
        help='Name (location) of the output file. The extension should be '
             '.json. If no extension is given, .json will be appended.')
    parse_conf.set_defaults(func=make_config)

    parse_clear = subs.add_parser('purge',
        help='Clean up a run directory. Removes any stub runs and re-numbers '
             'sub-folders to be sequential.')
    parse_clear.add_argument('dir', type=str,
        help='Run directory to purge. All runs that that don\'t have a '
             '.state or a saved model will be deleted.')
    parse_clear.set_defaults(func=purge)

    parse_desc = subs.add_parser('descriptions', aliases=['desc'],
        help='Prints the description field from the config file for each run.')
    parse_desc.add_argument('dir', type=str,
        help='Run directory to look in.')
    parse_desc.add_argument('-v', dest='verbose', action='store_true',
        help='Print verbose output.')
    parse_desc.set_defaults(func=desc)

    args = parser.parse_args()

    if 'func' in args:
        args.func(args)
    else:
        parser.print_help()

