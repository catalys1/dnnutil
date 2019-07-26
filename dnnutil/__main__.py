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


def compare(args):
    '''TODO:docs
    '''
    from collections import OrderedDict
    rundir = Path(args.dir)
    if args.runs is None:
        runs = sorted([x for x in rundir.iterdir() if x.stem.isnumeric()],
                      key=lambda x:int(x.name))
    else:
        def expand(x):
            if x.isnumeric():
                return x
            else:
                x1, x2 = x.split('-')
                return range(int(x1), int(x2) + 1)
        runs = []
        for run in args.runs:
            run = expand(run)
            if isinstance(run, str):
                runs.append(run)
            else:
                runs.extend([str(x) for x in run])
        runs = sorted([rundir / run for run in runs],
                      key=lambda x:int(x.name))

    if args.info is None:
        keysets = []
    else:
        keysets = [x.split('.') for x in args.info]

    info = []
    for run in runs:
        try:
            log = run / 'log.txt'
            acc = dnnutil.extract_metrics(log)[-1].max()
            run_info = OrderedDict(run=run.name, accuracy=acc)
            if args.info is not None:
                j = json.load(open(run / 'config.json'))
                for i in range(len(keysets)):
                    ks = keysets[i]
                    attribute = j[ks[0]]
                    for next_key in ks[1:]:
                        attribute = attribute.get(next_key, {})
                    attribute = attribute if attribute != {} else None
                    run_info[next_key] = attribute
            info.append(run_info)
        except FileNotFoundError as e:
            pass
        except ValueError as e:
            pass

    headers = list(info[0].keys())
    template = '  '.join(f'{{:<{len(x)}}}' for x in headers)
    print(template.format(*headers))
    for i in range(len(info)):
        vals = [str(info[i][x]) for x in headers]
        print(template.format(*vals))

    if args.plot:
        pass


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

    parse_comp = subs.add_parser('compare', aliases=['comp'],
        help='Display a comparison of best test accuracy between different '
             'runs in a run directory.')
    parse_comp.add_argument('dir', type=str,
        help='Run directory to look in.')
    parse_comp.add_argument('-r', '--runs', type=str, nargs='*',
        help='A list of runs to compare. If not specified, all runs in dir '
             'will be used. Supports ranges, like "1-5".')
    parse_comp.add_argument('-i', '--info', metavar='key', type=str, nargs='*',
        help='A set of attribute values to display. Each attribute consists '
             'of a period-seperated list of identifiers, i.e. "model.model" '
             'or "optimizer.kwargs.weight_decay"')
    parse_comp.add_argument('-p', '--plot', action='store_true',
        help='Plot the results on a bar graph.')
    parse_comp.set_defaults(func=compare)

    args = parser.parse_args()

    if 'func' in args:
        args.func(args)
    else:
        parser.print_help()

