Name: Gavin Zhao

Email: gz2133@nyu.edu

State of work:
The code works for the provided test cases. Cases where the pc or memory
overflowed were not tested.

Resources Used:
None.

Design Decisions:
My simulator from project 2 was tweaked to take in one extra argument:
information about the current state of the cache. The simulator would only take
this into account when processing a lw or sw argument.

When reading in command line arguments, the program either processes 3 or 6
integer arguments. This distinction decides whether 1 or 2 caches are created
using the create_cache() function. Caches are initialized to have numrows
* assoc blocks. Each block contains at least 2 pieces of information: the row
number and the tag. If the block is the first in its row, it also contains
a list that keeps track of the least recently used blocks in the row. The list
is initially [0,1,...,n] where n is the associativity. In the case of a miss,
the first element in the list is seen as the least recently used.

When the simulator reads in an lw instruction, the execute_lw() function is
called. It first calculates the row number, tag, and index of the first block
of the relevant row using the get_cache_info() function. The get_status()
function is then called, which returns True if the tag is found in the cache
row, and False otherwise. If there's a hit on L1, the LRU list is updated. If
there's a miss on L1, then L2 is checked, if there is one. If there's a hit on
L2, the least recently used block of L1 is evicted and the LRU list of L2 is
updated. If there's a miss on L2, the least recently used blocks of L1 and L2
are evicted. Otherwise, if there is a miss on L1, and L2 does not exist, then
the LRU block of L1 is evicted. Evictions are carried out using the evict()
function.

The execute_sw() function operates similarly, but checks both caches regardless
if the tag is found in L1 or not. If the tag is not found in a cache, the least
recently used block is evicted. If the tag is found, then the LRU list for the
row is updated.
