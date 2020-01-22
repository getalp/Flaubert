"""
Hang Le (hangtp.le@gmail.com)
"""
import sys
import unicodedata
import six
import string, re
import argparse
import gzip, bz2, json


def convert_to_unicode(text):
    """
    Converts `text` to Unicode (if it's not already), assuming UTF-8 input.
    """
    # six_ensure_text is copied from https://github.com/benjaminp/six
    def six_ensure_text(s, encoding='utf-8', errors='strict'):
        if isinstance(s, six.binary_type):
            return s.decode(encoding, errors)
        elif isinstance(s, six.text_type):
            return s
        else:
            raise TypeError("not expecting type '%s'" % type(s))

    return six_ensure_text(text, encoding="utf-8", errors="ignore")


def normalize_unicode(text):
    """
    Normalize unicode underlying representation
    """
    text = unicodedata.normalize("NFC", text)

    return text


def read_codepage(text, codepage='cp863'):
    """
    Keep only characters belonging to the character set of a language

    Args: 
    -- text: input text
    -- code page: for each language.
    Example: Code page 863 is the code page used to write French Canadian language.
    https://www.ascii-codes.com/cp863.html
    """
    text = text.encode(codepage, "ignore").decode(codepage)
    text = text.encode('utf-8').decode('utf-8')

    return text


def rm_spaces(text):
    """
    Remove multiple spaces
    """
    pattern = re.compile(r'( ){2,}')
    text = re.sub(pattern, r' ', text)

    return text


def process_url_html(text):
    """
    Remove URLs in text
    """
    pattern = re.compile(r'(?:www|http)\S+|<\S+|\w+\/*>')
    text = re.sub(pattern, '', text)

    return text


def cleaner(text, rm_new_lines=False, do_lower=False):
    """
    Clean up an input text
    """
    # Convert and normalize the unicode underlying representation
    text = convert_to_unicode(text)
    text = normalize_unicode(text)

    # Normalize whitespace characters and remove carriage return
    if rm_new_lines:
        remap = {ord('\f'):' ', ord('\r'): '', ord('\n'):'', ord('\t'):''}
        text = text.translate(remap)
    else:
        remap = {ord('\f'):' ', ord('\r'): ''}
        text = text.translate(remap)

    # Normalize URL links
    text = process_url_html(text)

    # remove multiple spaces in text
    text = rm_spaces(text)

    if do_lower:
        text = text.lower()

    return text


def run_cleaner():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--stdin', '-i', default=0, type=int, help='Flag if input from stdin')
    parser.add_argument('--path', '-p', default=None, type=str, help='Path to input text file')
    parser.add_argument('--rm_new_lines', '-r', default=0, type=str, help='Remove new lines')

    args = parser.parse_args()

    if args.stdin:
        for line in sys.stdin:
            line = cleaner(line.strip(' -='), rm_new_lines=args.rm_new_lines)
            print(u'%s' % line)

    else:
        if args.path.endswith('.json.gz'):
            with gzip.open(args.path, 'rt', encoding='utf-8') as f:
                for line in f:
                    json_line = json.loads(line)
                    if "index" not in json_line:
                        text = json_line["text"]
                        text = cleaner(re.sub(r'  \^ .*', '', text))
                        print(u'%s' % text)

        if args.path.endswith('.gz'):
            with gzip.open(args.path, 'rt', encoding='utf-8') as f:
                    for line in f:
                        line = cleaner(line.strip(' -='))
                        print(u'%s' % line)

        elif args.path.endswith('.bz2'):
            with bz2.open(args.path, 'rt', encoding='utf-8') as f:
                for line in f:
                    line = cleaner(line.strip(' -='))
                    print(u'%s' % line)

        else:
            with open(args.path, 'rt', encoding='utf-8') as f:
                for line in f:
                    line = cleaner(line.strip(' -='))
                    print(u'%s' % line)


if __name__ == "__main__":
    run_cleaner()