import gzip
import json
import re
from tqdm import tqdm

# Input: gzip'd jsonlines file containing Congressional speeches with (at least)
#   elements for chamber (House or Senate), party, and text of the speech.
#
# For each input line,
#   Grabs the json using json.loads()
#   If the chamber is the one we're interested in:
#      Extracts the party and text elements
#      Cleans the line of text by replacing runs of whitespace (\s+) with a single space
#      Adds cleaned line to list of lines
#      Adds party for this speech to list of parties
#   At end, returns two values: lines and parties
#     Note that these are parallel lists, i.e. parties[i] is the part for the legislator
#     who gave the speech in lines[i].
def read_and_clean_lines(infile, chamber='Senate'):
    print("\nReading and cleaning text from {}".format(infile))
    lines = []
    parties = []
    with gzip.open(infile,'rt') as f:
        for line in tqdm(f):
            j = json.loads(line)
            if (j['chamber'] == chamber):
                party      = j['party']
                text       = j['text']
                clean_text = re.sub(r"\s+"," ",text)
                lines.append(clean_text)
                parties.append(party)
    print("Read {} documents".format(len(lines)))
    print("Read {} labels".format(len(parties)))
    return lines, parties
