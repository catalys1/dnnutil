import dnnutil


def make_template(args):
    import os
    if os.path.isfile(args.file):
        while True:
            resp = input(f'{args.file} already exists. Overwrite it [y|N]? ')
            if resp.lower() in ['y', 'n']:
                break
        if resp == 'n':
            return

    with open(dnnutil.template, 'r') as f:
        temp = f.read()
    with open(args.file, 'w') as f:
        f.write(temp)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers()
    parse_new = subs.add_parser('new', aliases=['n'])
    parse_new.add_argument('-f', '--file', type=str, default='main.py',
        help='Name (location) of output file. Default: ./main.py')
    parse_new.set_defaults(func=make_template)
    args = parser.parse_args()

    if 'func' in args:
        args.func(args)
    else:
        parser.print_help()


