################################################################
##
## Demonstrations of CKY control structure
##
##   CKY minus the grammar. :)
## 
################################################################
import argparse

################################################################
# Convenience functions
################################################################
def ordinal(n):
    if n == 1:
        return("1st")
    elif n == 2:
        return("2nd")
    elif n == 3:
        return("3rd")
    else:
        return("{}th".format(n))

# Human-readable span
def span(words,start,end):
    return( "{}-{}({})".format(start,end," ".join(words[start:end])))

################################################################
# SLP version of CKY
################################################################
def cky_by_start_and_end(wordstring):
  words = wordstring.split()
  n     = len(words)

  print("CKY with the control structure in SLP figure 13.5\nCell [i,j] means word span from i to j\n")
  # for j from 1 to length(words)
  for j in range(1,n+1): 
      print("Looking for A's to put in cell [i={},j={}] span starting at {} and ending at {}, i.e. {}".format(j-1,j,j-1,j,span(words,j-1,j)))
      # for i from j-2 down to 0
      for i in range(j-2, -1, -1):
          print("  Looking for A's to put in cell [i={},j={}] (span starting at {} and ending at {}, i.e. {})".format(i,j,i,j,span(words,i,j)))
          # for k from i+1 to j-1
          for k in range(i+1,j-1+1):
              print("    Considering split point k={}, for combining B in [i={},j={}] with a C in [i={},j={}], i.e. {} + {}".format(k,i,k,k,j,span(words,i,k),span(words,k,j)))
  print("Checking for an S in cell [{},{}]".format(0,n))

################################################################
# Wikipedia/lecture version of CKY
################################################################
def cky_by_length_and_start(wordstring):
  words = wordstring.split()
  n     = len(words)

  print("CKY with control structure in the Wikipedia entry at https://en.wikipedia.org/wiki/CYK_algorithm")
  print("Cell [l,i] means span of length len=l starting at position i\n")

  # for j from 1 to length(words)
  for s in range(1,len(words)+1):
      print("Looking for A's to put in cell [len={},start={}], i.e. {}".format(1,s-1,span(words,s-1,s)))

  # for each l = 2 to n -- Length of span
  for l in range(2,len(words)+1):
      print("Building spans of length {} (row numbered {} in table)".format(l,l))
      # for each s = 1 to n-l+1 -- Start of span
      for i in range(0,n-l+1):
          print("  Looking for A's to put in cell [len={},start={}], i.e. {}".format(l,i,span(words,i,i+l)))
          # for each p = 1 to l-1 -- Partition of span
          for p in range(1,l-1+1):
              print("    Considering partition with p={} (split after {} word in the span), for combining B in [len={},start={}]  with C in [len={},start={}], i.e. {} + {}".format(p,ordinal(p),p,i,l-p,i+p,span(words,i,i+p),span(words,i+p,i+l)))
  print("Checking for an S in cell [len={},start={}]".format(n,0))



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Demonstrate CKY control structure')
    parser.add_argument('--version',     default='Wikipedia',            action='store', help="SLP or Wikipedia")
    parser.add_argument('--string',      default='this is a string',     action='store', help="String of tokens to parse, e.g. 'this is a test'")
    args = parser.parse_args()

    print("\nRunning CKY on: {}\n".format(args.string))
    
    if args.version == 'SLP':
        cky_by_start_and_end(args.string)
    elif args.version == 'Wikipedia':
        cky_by_length_and_start(args.string)
    else:
        print("Unknown value for --version")
