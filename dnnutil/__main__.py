import dnnutil
import os
from pathlib import Path


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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers()

    parse_new = subs.add_parser('new', aliases=['n'])
    parse_new.add_argument('-f', '--file', type=str, default='main.py',
        help='Name (location) of output file. Default: ./main.py')
    parse_new.set_defaults(func=make_template)

    parse_conf = subs.add_parser('conf', aliases=['c'])
    parse_conf.add_argument('file', type=str,
        help='Name (location) of the output file. The extension should be '
             '.json. If no extension is given, .json will be appended.')
    parse_conf.set_defaults(func=make_config)

    args = parser.parse_args()

    if 'func' in args:
        args.func(args)
    else:
        parser.print_help()


