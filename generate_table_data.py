#!/usr/bin/env python3
"""Generate fake table data."""

import string
from datetime import datetime, timedelta
from random import choice, randint, random, seed

from allometry.consts import ITIS_DIR, SEED

seed(SEED)

GUTTER = ' ' * 6  # Spaces between columns

UF_CHARS = string.ascii_uppercase + string.digits

HIGH = 9.999999
SUB = HIGH / 2.0
BEFORE_NEG = 3  # sign + before digits + point
BEFORE_POS = 2  # before digits + point

COLUMNS = {
    'OBS': {'width': 3, 'just': 'left'},
    'ID': {'width': 20, 'just': 'center'},
    'UF': {'width': 6, 'just': 'left'},
    'TFW': {'width': 6, 'just': 'right'},
    'SW': {'width': 6, 'just': 'left'},
    'WDV': {'width': 6, 'just': 'left'},
    'TBW': {'width': 8, 'just': 'left'},
    'USW': {'width': 6, 'just': 'left'},
    'PFW': {'width': 4, 'just': 'left'},
    'OCW': {'width': 4, 'just': 'left'},
    'AGW': {'width': 4, 'just': 'left'},
    'ASTLOG': {'width': 8, 'just': 'right'},
    'BIOLOG': {'width': 7, 'just': 'right'},
}

# Get some genus and species names to use as fake IDs
with open(ITIS_DIR / 'species.txt') as in_file:
    SPECIES = [ln.strip().upper() for ln in in_file.readlines()]

with open(ITIS_DIR / 'genera.txt') as in_file:
    GENERA = [ln.strip().upper() + ' SP' for ln in in_file.readlines()]


def fake_id(width=20):
    """Get a species or genera name for the species"""
    id_ = choice(SPECIES) if random() < 0.8 else choice(GENERA)
    return id_[:width].ljust(width)


def fake_uf(width=6):
    """Generate a fake UF ID.

    The UF ID is just a string of letters and digits between 5 and 6 characters long.
    """
    length = randint(5, width)
    uf = [choice(UF_CHARS) for _ in range(length + 1)]
    uf = ''.join(uf)
    return uf[:width].ljust(width)


def fake_float(after=4, neg=False):
    """Generate and format floats for the table data.

    The numbers are all similar with NA being represented as a lone decimal point.
    1 in 10 numbers will be NA.
    The numbers are returned as fixed length stings.
        after = The number of digits after the decimal point.
        neg = Does the number allow negative values?
    """
    num = random() * HIGH
    num = num if not neg else (num - SUB)

    if neg:
        formatter = f'{{: {BEFORE_NEG + after}.{after}f}}'
        na = f'  .{" " * after}'
    else:
        formatter = f'{{:{BEFORE_POS + after}.{after}f}}'
        na = f' .{" " * after}'

    formatted = formatter.format(num)

    # How missing numbers are reported
    if random() < 0.1:
        formatted = na

    return formatted


# ## Generate fake page text

# Generate a fake table.
#
# **Note that there will be other table formats.**

# In[12]:


MIN_ROWS = 50
MAX_ROWS = 76


def fake_data():
    """Generate the table body."""
    count = randint(MIN_ROWS, MAX_ROWS) + 1
    rows = []
    for i in range(1, count):
        rows.append({
            'OBS': str(i).rjust(3),
            'ID': fake_id(),
            'UF': fake_uf(),
            'TFW': fake_float(),
            'SW': fake_float(),
            'WDV': fake_float(),
            'TBW': fake_float(after=6),
            'USW': fake_float(),
            'PFW': fake_float(after=2),
            'OCW': fake_float(after=2),
            'AGW': fake_float(after=2),
            'ASTLOG': fake_float(after=5, neg=True),
            'BIOLOG': fake_float(neg=True),
        })
    return rows


def fake_lines(rows):
    """Add the rows first so we can easily calculate the page width in characters."""
    lines = []
    for row in rows:
        lines.append((GUTTER.join(v for v in row.values())))
    return lines


def fake_header(line_len):
    """Generate fake page header."""
    header = ' '.join(list('STATISTICAL ANANYSIS SYSTEM'))
    header = header.center(line_len)
    page_no = randint(1, 9)
    header += '  ' + str(page_no)
    return header, page_no


def fake_date(line_len):
    """Generate a fake date line."""
    date = datetime(randint(1970, 1990), 1, 1)
    date += timedelta(days=randint(0, 365))
    date += timedelta(seconds=randint(0, 24 * 60 * 60))
    date = date.strftime('%H:%M %A, %B %m, %Y')
    date_line = date.rjust(line_len + 3) + '\n'
    return date_line, date


def fake_column_headers():
    """Generate fake column headers."""
    headers = []
    for name, data in COLUMNS.items():
        width = data['width']
        just = data['just']

        if just == 'center':
            header = name.center(width)
        elif just == 'right':
            header = name.rjust(width)
        else:
            header = name.ljust(width)

        headers.append(header[:width])

    column_headers = GUTTER.join(h for h in headers) + '\n'
    return column_headers


# def generate_images(args):
#     """Generate the images for the pages."""
#     pages = list(Path(args.text_dir).glob('*.json'))
#     if args.count and args.count < len(pages):
#         pages = pages[:args.count]
#
#     for i in tqdm(pages):
#         font_file = choice(FONTS)
#
#         font_name = splitext(basename(font_file))[0]
#         base_name = f'table_{str(i).zfill(3)}'
#
#         params = FONT_PARAMS.get(font_name, {})
#         clean_params = {
#             'font': font_file,
#             'font_size': params.get('size', 36),
#         }
#         dirty_params = {
#             'snow_fract': params.get('snow', 0.05),
#             'image_filter': params.get('filter', 'max'),
#         }
#
#         data = build_page(
#             args, base_name,
#             clean_params=clean_params,
#             dirty_params=dirty_params)
#
#         data['clean'] = clean_params
#         data['dirty'] = dirty_params
#
#         with open(TEXT_DIR / (base_name + '.json'), 'w') as json_file:
#             json.dump(data, json_file, indent='  ')


def fake_page():
    """Create a fake table."""
    rows = fake_data()
    lines = fake_lines(rows)
    line_len = len(lines[0])
    page_header, page_no = fake_header(line_len)
    date_line, date = fake_date(line_len)
    column_headers = fake_column_headers()

    page = [page_header, date_line, column_headers] + lines
    page = '\n'.join(page) + '\n'

    data = {
        'rows': rows,
        'page_no': page_no,
        'date': date,
        'page': page,
    }

    return page, data
