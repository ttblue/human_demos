"""
add terminal color codes to strings.
Taken from John's rapprentice.
"""

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)
    
    
def colorize(string, color, bold=False, highlight = False):
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

def redprint(text, bold=True):
    print colorize(text,'red',bold)
def blueprint(text, bold=True):
    print colorize(text,'blue',bold)
def greenprint(text, bold=True):
    print colorize(text,'green',bold)
def yellowprint(text, bold=True):
    print colorize(text,'yellow',bold)