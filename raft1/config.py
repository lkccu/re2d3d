from argparse import Namespace

args = Namespace(a=1, b='c')
def RAFTConfig(dropout,alternate_corr,small,mixed_precision):
    args = Namespace(dropout=dropout, alternate_corr=alternate_corr,small=small,mixed_precision=mixed_precision)
    return args