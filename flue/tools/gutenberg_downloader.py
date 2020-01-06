"""
Hang Le (hangtp.le@gmail.com)
"""

import os
import re
import gzip
import argparse
import json

import requests, bs4, zipfile, glob
from urllib.request import urlretrieve


def check_url(url):
    """
    Check if an url exists
    """

    request = requests.get(url)
    if request.status_code == 200:
        return True
    else:
        return False


def dir2ids(indir):
    """
    Get a list of valid book ids from an input directory

    Returns: list of book ids
    """
    # Get files in directory
    file_list = os.listdir(indir)
    # Get book ids from file names
    book_ids = [re.findall('[0-9]+', t) for t in file_list]
    book_ids = [t for sublist in book_ids for t in sublist]

    return book_ids


def id2url(book_id, encoding='utf-8'):
    """
    Returns corresponding url to download from an input book id

    Args: 
        ::book_id:: book ID number
        ::encoding:: encoding type preferred to get

    Check https://www.gutenberg.org/files/ for details about files' extensions:
    - .txt is plaintext US-ascii
    - -8.txt is 8-bit plaintext, multiple encodings
    - -0.txt is UTF-8
    """
    # Get the priority tuple of file types
    ascii_first = ('.txt', '-0.txt', '-8.txt')
    utf8_first = ('-0.txt', '-8.txt', '.txt')
    extensions = utf8_first if encoding=='utf-8' else ascii_first

    # Get the subdirectory of urls
    mirror = os.environ.get('GUTENBERG_MIRROR', 'http://aleph.gutenberg.org')
    root = mirror.strip().rstrip('/')
    subdir_path = id2subdir(book_id)
    # print('subdir_path=', subdir_path)

    # Iterate over extension and save files according to type preference
    for extension in extensions:
        url = '{root}/{path}/{book_id}{extension}'.format(root=root,
                                                            path=subdir_path,
                                                            book_id=book_id,
                                                            extension=extension)
        # print('url=', url)
        if check_url(url):
            return url


def id2subdir(book_id):
    """
    Returns subdirectory that a book_id will be found in a gutenberg mirror.
    """

    str_book_id = str(book_id).zfill(2)
    all_but_last_digit = list(str_book_id[:-1])
    subdir_part = "/".join(all_but_last_digit)
    subdir = "{}/{}".format(subdir_part, book_id)

    return subdir  


def get_book_urls(indir, page_book_urls, update_url):
    """
    Returns a list of unique book ids
    """

    # Create a directory to save files if not already exists
    if not os.path.exists(indir):
        os.makedirs(indir)

    # Get the list of urls
    book_urls = []

    if os.path.exists(os.path.join(indir, 'gutenberg_book_urls.txt')):
        if not update_url:
            print('Loading URLs txt file')
            # Reading from txt files
            with open(os.path.join(indir, 'gutenberg_book_urls.txt'), 'r') as f:
                for line in f:
                    line = line.replace("\n", "")
                    book_urls.append(line)

        elif update_url:
            print('Removing URLs txt file')
            # remove old file
            os.remove(os.path.join(indir, 'gutenberg_book_urls.txt'))
            print('Finished removing URLs txt file')

    if not os.path.exists(os.path.join(indir, 'gutenberg_book_urls.txt')):

        while True:

            is_last_page = False
            print('Reading page: ' + page_book_urls)

            # Get the urls
            page_w_books = requests.get(page_book_urls, timeout=20.0)

            if page_w_books:
                page_w_books = bs4.BeautifulSoup(page_w_books.text, "lxml")
                urls = [el.get('href') for el in page_w_books.select('body > p > a[href^="http://aleph.gutenberg.org/"]')]
                url_to_next_page = page_w_books.find_all('a', string='Next Page')

                if len(urls) > 0:
                    book_urls.append(urls)

                    if url_to_next_page[0]:
                        page_book_urls = "http://www.gutenberg.org/robot/" + url_to_next_page[0].get('href')
                else:
                    is_last_page = True

            if is_last_page:
                break

        book_urls = [item for sublist in book_urls for item in sublist]

        # Backing up the list of URLs
        with open(os.path.join(indir, 'gutenberg_book_urls.txt'), 'w') as output:
            for u in book_urls:
                output.write('%s\n' % u)


    return book_urls


def get_book_ids(book_urls):
    """
    Return a list of unique book ids from a list of book urls
    """

    # Check if books are duplicated
    unique_list = {}
    num_processed = 0

    for url in book_urls:
        # print('url=', url)
        all_nums = re.findall('[0-9]+', url)
        # if all_nums[-1] == all_nums[-2] or all_nums[-2] == all_nums[-3]:
        unique_list[all_nums[-2]] = unique_list.get(all_nums[-2], 0) + 1
        num_processed += 1

    # Get the list of unique book ids
    book_ids = list(unique_list.keys())

    print('Number of links before removing duplicates:', len(book_urls))
    print('Number of links processed:', num_processed)
    print('Number of unique books retained:', len(unique_list.keys()))

    return book_ids


def download_books(book_ids, indir):
    """
    Download books from a list of book ids and save all to output_dir
    """

    # Create directory if not already exists
    if not os.path.exists(indir):
        os.makedirs(indir)

    # Get list of book ids
    available_books = dir2ids(indir)

    count = 0
    # Download book
    for book_id in book_ids:
        
        # Check if a book is already downloaded
        if book_id not in available_books:
            # Download books that are not downloaded yet
            print('Downloading and saving book ID no.'+ book_id + '...')
            file_path = os.path.join(indir,'{}.txt'.format(book_id))
            download_url = id2url(book_id)
            # print('download_url=', download_url)

            response = requests.get(download_url)

            # Overwrite the encoding
            if response.encoding != response.apparent_encoding:
                response.encoding = response.apparent_encoding
            text = response.text

            with open(file_path, 'wb') as f:
                f.write(text.encode('utf-8'))

            count += 1
    
    print('Number of books downloaded and saved:', count)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("-indir", default=None, type=str, required=True, 
                        help='Path to directory to save downloaded files')
    parser.add_argument('-lang', default='fr', type=str, required=False,
                        help='Language to download')
    parser.add_argument('-update_url', default=1, type=int, required=False,
                        help='Choose to update book URLs if necessary (1/0)')

    args = parser.parse_args()

    print('-' * 100)
    print('Getting book URLs...')
    # Get list of book_urls
    page_book_urls = 'http://www.gutenberg.org/robot/harvest?filetypes[]=txt&langs[]={lang}'.format(lang=args.lang)
    book_urls = get_book_urls(args.indir, page_book_urls, args.update_url)
    
    # Get unique book_ids
    print('-' * 100)
    print('Getting book IDs...')
    book_ids = get_book_ids(book_urls)
    print('-' * 100)

    # Download and save book to indir
    print('Downloading data...')
    download_books(book_ids, args.indir)
    print('-' * 100)


if __name__ == '__main__':
    main()