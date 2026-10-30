content = open('generate_project_guide_pdf.py', encoding='utf-8').read()

replacements = [
    ('\u2014', '--'),
    ('\u2013', '-'),
    ('\u2018', "'"),
    ('\u2019', "'"),
    ('\u201c', '"'),
    ('\u201d', '"'),
    ('\u2022', '-'),
    ('\u2190', '<-'),
    ('\u2192', '->'),
    ('\u2264', '<='),
    ('\u2265', '>='),
    ('\u00b7', '-'),
    ('\u0394', 'Delta'),
    ('\u03b1', 'alpha'),
    ('\u00e9', 'e'),
    (chr(149), '-'),
    ('\u2605', '*'),
    ('\u00d7', 'x'),
]

for old, new in replacements:
    content = content.replace(old, new)

# Also encode to latin-1 safe
safe = []
for ch in content:
    try:
        ch.encode('latin-1')
        safe.append(ch)
    except UnicodeEncodeError:
        safe.append('?')
content = ''.join(safe)

open('generate_project_guide_pdf.py', 'w', encoding='utf-8').write(content)
print('Done - unicode fixed')
