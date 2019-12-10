"""
Hang Le (hangtp.le@gmail.com)

Part of the script, including TEXT_START_MARKERS, TEXT_END_MARKERS and strip_headers function
is copied from https://github.com/kiasar/gutenberg_cleaner/blob/master/_cleaning_options/strip_headers.py.
"""

import os
import json
import string
import re
from nltk import word_tokenize, sent_tokenize
from gutenberg_downloader import *

TEXT_START_MARKERS = frozenset((
    "*END*THE SMALL PRINT",
    "*** START OF THE PROJECT GUTENBERG",
    "*** START OF THIS PROJECT GUTENBERG",
    "The Project Gutenberg EBook of",
    "This eBook is for the use of",
    "You may copy it, give it away or re-use it under the terms of the Project Gutenberg",
    "This file was produced from images generously made available",
    "by the Bibliothèque nationale de France (BnF/Gallica) at",
    "file was produced from images generously",
    "Bibliotheque nationale de France ( BnF / Gallica )",
    "http://gallica.bnf.fr.",
    "This etext was prepared by",
    "E-text prepared by",
    "Produced by",
    "Distributed Proofreading Team",
    "Proofreading Team at http://www.pgdp.net",
    "http://gallica.bnf.fr)",
    "      http://archive.org/details/",
    "http://www.pgdp.net",
    "by The Internet Archive)",
    "by The Internet Archive/Canadian Libraries",
    "by The Internet Archive/American Libraries",
    "public domain material from the Internet Archive",
    "Internet Archive)",
    "Internet Archive/Canadian Libraries",
    "Internet Archive/American Libraries",
    "material from the Google Print project",
    "*END THE SMALL PRINT",
    "***START OF THE PROJECT GUTENBERG",
    "This etext was produced by",
    "*** START OF THE COPYRIGHTED",
    "The Project Gutenberg",
    "http://gutenberg.spiegel.de/ erreichbar.",
    "Project Runeberg publishes",
    "Beginning of this Project Gutenberg",
    "Project Gutenberg Online Distributed",
    "Gutenberg Online Distributed",
    "the Project Gutenberg Online Distributed",
    "Project Gutenberg TEI",
    "This eBook was prepared by",
    "http://gutenberg2000.de erreichbar.",
    "This Etext was prepared by",
    "This Project Gutenberg Etext was prepared by",
    "Gutenberg Distributed Proofreaders",
    "Project Gutenberg Distributed Proofreaders",
    "the Project Gutenberg Online Distributed Proofreading Team",
    "**The Project Gutenberg",
    "*SMALL PRINT!",
    "More information about this book is at the top of this file.",
    "tells you about restrictions in how the file may be used.",
    "l'authorization à les utilizer pour preparer ce texte.",
    "of the etext through OCR.",
    "*****These eBooks Were Prepared By Thousands of Volunteers!*****",
    "We need your donations more than ever!",
    " *** START OF THIS PROJECT GUTENBERG",
    "****     SMALL PRINT!",
    '["Small Print" V.',
    '      (http://www.ibiblio.org/gutenberg/',
    'and the Project Gutenberg Online Distributed Proofreading Team',
    'Mary Meehan, and the Project Gutenberg Online Distributed Proofreading',
    '                this Project Gutenberg edition.',
    "Proofreaders.",
))


TEXT_END_MARKERS = frozenset((
    "*** END OF THE PROJECT GUTENBERG",
    "*** END OF THIS PROJECT GUTENBERG",
    "***END OF THE PROJECT GUTENBERG",
    "End of the Project Gutenberg",
    "End of The Project Gutenberg",
    "Ende dieses Project Gutenberg",
    "by Project Gutenberg",
    "End of Project Gutenberg",
    "End of this Project Gutenberg",
    "Ende dieses Projekt Gutenberg",
    "        ***END OF THE PROJECT GUTENBERG",
    "*** END OF THE COPYRIGHTED",
    "End of this is COPYRIGHTED",
    "***** This file should be named",
    "This and all associated files",
    "Ende dieses Etextes ",
    "Ende dieses Project Gutenber",
    "Ende diese Project Gutenberg",
    "**This is a COPYRIGHTED Project Gutenberg Etext, Details Above**",
    "Fin de Project Gutenberg",
    "The Project Gutenberg Etext of ",
    "Ce document fut presente en lecture",
    "Ce document fut présenté en lecture",
    "More information about this book is at the top of this file.",
    "We need your donations more than ever!",
    "END OF PROJECT GUTENBERG",
    " End of the Project Gutenberg",
    " *** END OF THIS PROJECT GUTENBERG",
    "Updated editions will replace the previous one",
    "Creating the works from public domain print editions means that"
))


LEGALESE_START_MARKERS = frozenset(("<<THIS ELECTRONIC VERSION OF", "*** START: FULL LICENSE",
                                    "THE FULL PROJECT GUTENBERG LICENSE",
                                    "PLEASE READ THIS BEFORE YOU"))


LEGALESE_END_MARKERS = frozenset(("SERVICE THAT CHARGES FOR DOWNLOAD",))


def strip_headers(text):
    """
    Remove lines that are part of the Project Gutenberg header or footer.
    Note: this function is a port of the C++ utility by Johannes Krugel. The
    original version of the code can be found at:
    http://www14.in.tum.de/spp1307/src/strip_headers.cpp

    reference: https://github.com/kiasar/gutenberg_cleaner/tree/master/_cleaning_options
    
    Args:
        text (unicode): The body of the text to clean up.
    Returns:
        unicode: The text with any non-text content removed.
    """
    lines = text.splitlines()
    sep = str(os.linesep)

    out = []
    i = 0
    footer_found = False
    ignore_section = False

    for line in lines:
        reset = False

        if i <= 600:
            # Check if the header ends here
            if any(line.startswith(token) for token in TEXT_START_MARKERS):
                reset = True

            # If it's the end of the header, delete the output produced so far.
            # May be done several times, if multiple lines occur indicating the
            # end of the header
            if reset:
                out = []
                continue

        if i >= 100:
            # Check if the footer begins here
            if any(line.startswith(token) for token in TEXT_END_MARKERS):
                footer_found = True

            # If it's the beginning of the footer, stop output
            if footer_found:
                break

        if any(line.startswith(token) for token in LEGALESE_START_MARKERS):
            ignore_section = True
            continue
        elif any(line.startswith(token) for token in LEGALESE_END_MARKERS):
            ignore_section = False
            continue

        if not ignore_section:
            out.append(line.rstrip(sep))
            i += 1

    return sep.join(out)


def save_to_txt(indir, outdir):
    """
    Clean txt files in indir and save clean file to output directory
    """
    # Create directory if not already exists
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Get the list of book ids from download folder
    book_ids = dir2ids(indir)
    file_list = [str(t) + '.txt' for t in book_ids]

    # Get the list of book ids already cleaned
    book_ids_cleaned = dir2ids(outdir)
    book_ids_cleaned = [str(t) + '.txt' for t in book_ids_cleaned]

    count = 1
    for txt_file in file_list:
        if txt_file not in book_ids_cleaned:
            print('Cleaning and saving book', count, '...')
            # Open raw txt file
            with open(os.path.join(indir, txt_file), encoding='utf8', mode='r') as f:
                # Strip the headers, footers, legal copyright
                text = f.read()
                # Remove newlines characters
                pattern = re.compile('(\\n){2,}')
                text = re.sub(pattern, '_jsqmlf_', text)

                pattern = re.compile('\\n')
                text = re.sub(pattern, ' ', text)

                pattern = re.compile('_jsqmlf_')
                text = re.sub(pattern, '\\n', text)
                    
                # Strip headers
                text = strip_headers(text)
                sent_list = sent_tokenize(text)

            with open(os.path.join(outdir, txt_file), encoding='utf8', mode='w') as f:
                for sent in sent_list:
                    f.write(sent)
                    f.write('\n')
            
            count += 1

    print('Number of books cleaned and saved:', count-1)


def save_to_json(indir, outdir):
    """
    Save clean txt files to json format
    """
    json_metadata = {}

    # Create directory if not already exists
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Get list of books downloaded
    book_ids = dir2ids(indir)
    print('Number of books=', len(book_ids))

    count = 1
    with open(os.path.join(outdir, 'gutenberg.json'), 'w', encoding='utf8') as out:

        for book_id in book_ids:
            print(str(count) + '.', 'Cleaning book', book_id , '...')
            # Add book id to dictionary
            json_metadata['id'] = book_id
            # Get url
            json_metadata['url'] = id2url(book_id)

            # Strip the headers, footers, legal copyright
            with open(os.path.join(indir, str(book_id)+'.txt'), encoding='utf8', mode='r') as f:
                # Get text
                text = strip_headers(f.read())

                for line in text.split('\n'):
                    if (line != '\n') and (line != ' ') and (line != ''):
                        json_metadata['title'] = line
                        break

                json_metadata['text'] = text

            if count == 1:
                out.write('[')

            # Save json files
            out_str = json.dumps(json_metadata, ensure_ascii=False)
            out.write(out_str)
            if count != len(book_ids):
                out.write(',')
            else:
                out.write(']')

            count += 1

    print('Number of books cleaned and saved:', count-1)


def main_gutenberg_cleaner():

    parser = argparse.ArgumentParser()

    parser.add_argument("-indir", default=None, type=str, required=True, 
                        help='Path to directory to save downloaded files')
    parser.add_argument('-outdir', default=None, type=str, required=True,
                        help='Path to directory to save clean files')

    parser.add_argument('-json', default=0, type=int, required=False,
                        help='Save file to json format or not (1/0)')

    args = parser.parse_args()

    
    if args.json:
        print('Cleaning and saving to json file...')
        # Save metadata in json format
        save_to_json(args.indir, args.outdir)

    else:
        print('Cleaning and saving to txt files...')
        # Save clean text to txt files
        save_to_txt(args.indir, args.outdir)


if __name__ == '__main__':
    main_gutenberg_cleaner()