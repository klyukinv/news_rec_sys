# -*- coding: utf-8 -*-
import click
import logging
import gzip
import shutil

from pathlib import Path
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    filenames = ['items.json.gz', 'train.json.gz']
    for filename_arch in filenames:
        with gzip.open(input_filepath + '/' + filename_arch, 'rb') as f_in:
            filename_new = output_filepath + '/' + '.'.join(filename_arch.split('.')[:-1])
            with open(filename_new, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
