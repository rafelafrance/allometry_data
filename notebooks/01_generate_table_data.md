# Test Creating Fake Allometry Data

Please note that this is just an experiment and is not expected to work in the future or be used directly.


```python
import json
import re
import string
from collections import defaultdict, deque
from datetime import datetime, timedelta
from functools import partial
from random import choice, randint, random, randrange, seed
from os.path import basename, splitext
from pathlib import Path

import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFilter, ImageFont
```


```python
DATA_DIR = Path('..') / 'data'
ITIS_DIR = DATA_DIR / 'itis'
CLEAN_DIR = DATA_DIR / 'clean'
DIRTY_DIR = DATA_DIR / 'dirty'
TEXT_DIR = DATA_DIR / 'text'

FONTS_DIR = Path('..') / 'fonts'
```


```python
SEED = 981
seed(SEED)
```


```python
GUTTER = ' ' * 6  # Spaces between columns
```

Get a list of all of the fonts we will be using.


```python
FONTS = FONTS_DIR.glob('*/*.ttf')
FONTS = sorted([str(f) for f in FONTS])
```

## Columns for the fake data

Get some genus and species names to use as fake IDs


```python
with open(ITIS_DIR / 'species.txt') as in_file:
    SPECIES = [ln.strip().upper() for ln in in_file.readlines()]

with open(ITIS_DIR / 'genera.txt') as in_file:
    GENERA = [ln.strip().upper() + ' SP' for ln in in_file.readlines()]
```


```python
def fake_id(width=20):
    id_ = choice(SPECIES) if random() < 0.8 else choice(GENERA)
    return id_[:width].ljust(width)
```

The UF ID is just a string of letters and digits between 5 and 6 characters long.


```python
UF_CHARS = string.ascii_uppercase + string.digits
```


```python
def fake_uf(width=6):
    length = randint(5, width)
    uf = [choice(UF_CHARS) for i in range(length + 1)]
    uf = ''.join(uf)
    return uf[:width].ljust(width)
```

The numbers are all similar with NA being represented as a lone decimal point. 1 in 10 numbers will be NA. The numbers are returned as fixed length stings.

after = The number of digits after the decimal point.

neg = Does the number allow negative values?


```python
HIGH = 9.999999
SUB = HIGH / 2.0
BEFORE_NEG = 3  # sign + before digits + point
BEFORE_POS = 2  # before digits + point
```


```python
def fake_float(after=4, neg=False):
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
```

## Generate fake page text

Generate a fake table.

**Note that there will be other table formats.**


```python
MIN_ROWS = 50
MAX_ROWS = 76
```


```python
def fake_data():
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
```

Add the rows first so we can easily calculate the page width in characters.


```python
def fake_lines(rows):
    lines = []
    for row in rows:
        lines.append((GUTTER.join(v for v in row.values())))
    return lines
```

Generate fake page header.


```python
def fake_header(line_len):
    header = ' '.join(list('STATISTICAL ANANYSIS SYSTEM'))
    header = header.center(line_len)
    page_no = randint(1, 9)
    header += '  ' + str(page_no)
    return header, page_no
```

Generate a fake date line


```python
def fake_date(line_len):
    date = datetime(randint(1970, 1990), 1, 1)
    date += timedelta(days=randint(0, 365))
    date += timedelta(seconds=randint(0, 24 * 60 * 60))
    date = date.strftime('%H:%M %A, %B %m, %Y')
    date_line = date.rjust(line_len + 3) + '\n'
    return date_line, date
```

Generate fake column headers.


```python
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
```


```python
def fake_column_headers():
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
```

Generate the page.


```python
def fake_page():
    rows = fake_data()
    lines = fake_lines(rows)
    line_len = len(lines[0])
    page_header, page_no = fake_header(line_len)
    date_line, date = fake_date(line_len)
    column_headers = fake_column_headers()

    page = [page_header, date_line, column_headers] + lines
    page = page = '\n'.join(page) + '\n'

    data = {
        'rows': rows,
        'page_no': page_no,
        'date': date,
        'page': page,
    }

    return page, data
```

Randomly translate the image left/right and up/down.


```python
def translate(image, text, up_down=None, left_right=None):
    x = int((WIDTH - size[0]) / 4)
    y = int((HEIGHT - size[1]) / 4)

    if random() < 0.5:
        x = -x
    if random() < 0.5:
        y = -y

    image.translate()
    
    return image
```

## Generate clean fake page image


```python
WIDTH = 4500
HEIGHT = 3440
```


```python
def clean_image(page, clean_params):
    font = clean_params['font']
    font_size = clean_params['font_size']

    font = ImageFont.truetype(font=font, size=font_size)
    
    size = font.getsize_multiline(page)
    clean_params['text_size'] = list(size)

    image = Image.new(mode='L', size=(WIDTH, HEIGHT), color='white')

    draw = ImageDraw.Draw(image)
    
    dx = WIDTH - size[0]
    dy = HEIGHT - size[1]

    x = (dx // 2) + (randint(0, dx // 4) * choice([1, -1]))
    y = (dy // 2) + (randint(0, dy // 4) * choice([1, -1]))

    draw.text((x, y), page, font=font, fill='black')

    return image, size
```

## Dirty the image

A function to change random values in a numpy array.


```python
def add_snow(data, dirty_params, low=128, high=255):
    snow_fract = dirty_params['snow_fract']
    dirty_params['snow_low'] = low
    dirty_params['snow_high'] = high

    shape = data.shape
    data = data.flatten()
    how_many = int(data.size * snow_fract)
    mask = np.random.choice(data.size, how_many)
    data[mask] = np.random.randint(low, high)
    data = data.reshape(shape)
    return data
```

Filter the image to make the snow more destructive.


```python
def filter_image(image, dirty_params):
    image = image.filter(ImageFilter.UnsharpMask())

    image_filter = dirty_params.get('filter')

    if image_filter == 'max':
        image = image.filter(ImageFilter.MaxFilter())
    elif image_filter == 'median':
        image = image.filter(ImageFilter.MedianFilter())
    elif image_filter == 'mode':
        image = image.filter(ImageFilter.ModeFilter())

    return image
```

Rotate the image slightly.


```python
def rotate_image(image, dirty_params):
    if (theta := dirty_params.get('theta')) is None:
        if random() < 0.5:
            theta = randrange(0.0, 2.0, 1.0)
        else:
            theta = randrange(358.0, 360.0, 1.0)
        dirty_params['theta'] = theta

    return image.rotate(theta)
```

Dirty the image randomly.


```python
def dirty_image(image, dirty_params):
    dirty = np.array(image)
    dirty = add_snow(dirty, dirty_params)
    dirty = Image.fromarray(dirty)

    dirty = filter_image(dirty, dirty_params)

    dirty = rotate_image(dirty, dirty_params)

    dirty = dirty.convert('RGB')

    return dirty
```

## Put it all together


```python
def build_page(base_name, clean_params, dirty_params):
    name = base_name + '.jpg'

    page, data = fake_page()

    clean, size = clean_image(page, clean_params)
    clean.save(CLEAN_DIR / name, 'JPEG')

    dirty = dirty_image(clean, dirty_params)
    dirty.save(DIRTY_DIR / name, 'JPEG')

    return data
```

Fonts sometimes require special treatment


```python
FONT_PARAMS = {
    'B612Mono-Bold': {},
    'B612Mono-BoldItalic': {},
    'B612Mono-Italic': {'filter': 'median'},
    'B612Mono-Regular': {'filter': 'median'},
    'CourierPrime-Bold': {},
    'CourierPrime-BoldItalic': {},
    'CourierPrime-Italic': {'filter': 'median'},
    'CourierPrime-Regular': {'filter': 'median'},
    'CutiveMono-Regular': {'filter': 'median'},
    'RobotoMono-Italic-VariableFont_wght': {'filter': 'median', 'size': 32},
    'RobotoMono-VariableFont_wght': {'filter': 'median', 'size': 32},
    'SyneMono-Regular': {'filter': 'median'},
    'VT323-Regular': {'filter': 'median'},
    'XanhMono-Italic': {'filter': 'median'},
    'XanhMono-Regular': {'filter': 'median'},
    'Kingthings_Trypewriter_2': {'filter': 'median'},
    'OCRB_Medium': {'filter': 'median'},
    'OCRB_Regular': {},
    'OcrB2': {},
}
```


```python
def create_pages():
    font_file = choice(FONTS)

    font_name = splitext(basename(font_file))[0]
    base_name = f'table_{str(i).zfill(3)}'

    params = FONT_PARAMS.get(font_name, {})
    clean_params = {
        'font': font_file,
        'font_size': params.get('size', 36),
    }
    dirty_params = {
        'snow_fract': params.get('snow', 0.05),
        'image_filter': params.get('filter', 'max'),
    }

    data = build_page(
        base_name,
        clean_params=clean_params,
        dirty_params=dirty_params)

    data['clean'] = clean_params
    data['dirty'] = dirty_params

    with open(TEXT_DIR / (base_name + '.json'), 'w') as json_file:
        json.dump(data, json_file, indent='  ')


# for i in tqdm(range(100)):
#     create_pages()
#     pass
```

    100%|██████████| 100/100 [00:00<00:00, 1800130.47it/s]



```python

```
