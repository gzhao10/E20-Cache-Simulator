#!/usr/bin/python3

"""
CS-UY 2214
Gavin Zhao
E20 cache simulator
simcache.py
"""

from collections import namedtuple
import re
import argparse

Constants = namedtuple("Constants",["NUM_REGS", "MEM_SIZE", "REG_SIZE"])
constants = Constants(NUM_REGS = 8,
                      MEM_SIZE = 2**13,
                      REG_SIZE = 2**16)

def create_cache(numrows, assoc):
    """
    Creates a cache with numrows * assoc entries.
    Each entry stores the row number and the tag.
    The first block of each row also stores the LRU list.
    Parameters:
        numrows: int = number of rows in the cache
        assoc: int = associativity of the cache
    """
    #each entry stores row #, tag, LRU
    cache = [[None] * 3 for i in range(numrows * assoc)]

    for i in range(len(cache)):
        cache[i][0] = i//assoc   #row number
        if i % assoc == 0:       #LRU list only created for first block per row
            cache[i][2] = [i for i in range(assoc)] #lru is [0,1,...] initially
    return cache


def execute_lw(addr, pc, info):
    """
    Executes a lw instruction, given the address, pc, and current state
    of the caches.
    One cache:
        The address is searched for in L1. If there is a hit, the LRU gets
        updated. If there is a miss, then the first slot in the LRU list
        is evicted, and the LRU is updated accordingly.
    Two caches:
        The address is searched for in L1. If there is a hit, the L1 LRU is
        updated. If there is a miss, then the first slot in the L1 LRU list is
        evicted and the address is searched for in L2. If there is a hit in
        L2, the L2 LRU is updated. If there is a miss, the first slot in the
        L2 LRU list is evicted.
    The log entries of the caches (if they are accessed) are printed.
    Parameters:
        addr: int = memory address being accessed
        pc: int = program counter of the memory access instruction
        info: list = contains current state of the cache, the number of rows,
                     associativity, and blocksize for L1 (and L2 if there is one)
    """
    #Find the row, tag of the address. Also find the first block of the row
    #(where the LRU list is located in the cache)
    L1, L1numrows, L1assoc, L1blocksize = info[0], info[1], info[2], info[3]
    L1row, L1tag, L1_first_block = get_cache_info(addr, L1numrows, L1assoc, L1blocksize)
    #Is it a hit or miss?
    L1status = get_status(L1, L1_first_block, L1tag, L1assoc)
    if L1status:
        print_log_entry('L1', 'HIT', pc, addr, L1row)
    else:
        print_log_entry('L1', 'MISS', pc, addr, L1row)

    #if L1 miss, check L2, if there is one
    if not L1status and len(info) > 4:
        L2, L2numrows, L2assoc, L2blocksize = info[4], info[5], info[6], info[7]
        L2row, L2tag, L2_first_block = get_cache_info(addr, L2numrows, L2assoc, L2blocksize)
        L2status = get_status(L2, L2_first_block, L2tag, L2assoc)

        #if L2 hit, evict from L1
        if L2status:
            evict(L1, L1_first_block, L1tag)

        #if L2 miss, evict from L2 and L1
        else:
            evict(L2, L2_first_block, L2tag)
            evict(L1, L1_first_block, L1tag)

        if L2status:
            print_log_entry('L2', 'HIT', pc, addr, L2row)
        else:
            print_log_entry('L2', 'MISS', pc, addr, L2row)

    #if only L1 exists, and L1 miss, evict from L1
    elif not L1status:
        evict(L1, L1_first_block, L1tag)


def execute_sw(addr, pc, info):
    """
    Calls writes to L1 and L2 (if there is one).
    Prints log entries.
    Parameters:
        addr: int = memory address being accessed
        pc: int = program counter of the memory access instruction
        info: list = contains current state of the cache, the number of rows,
                     associativity, and blocksize for L1 (and L2 if there is one)
    """
    #Find the row, tag of the address. Also find the first block of the row
    #(where the LRU list is located in the cache)
    L1, L1numrows, L1assoc, L1blocksize = info[0], info[1], info[2], info[3]
    L1row, L1tag, L1_first_block = get_cache_info(addr, L1numrows, L1assoc, L1blocksize)
    print_log_entry('L1', 'SW', pc, addr, L1row)
    #Write to L1
    L1status = get_status(L1, L1_first_block, L1tag, L1assoc)
    if not L1status:
        evict(L1, L1_first_block, L1tag)

    #Do the same process if L2 exists
    if len(info) > 4:
        L2, L2numrows, L2assoc, L2blocksize = info[4], info[5], info[6], info[7]
        L2row, L2tag, L2_first_block = get_cache_info(addr, L2numrows, L2assoc, L2blocksize)
        print_log_entry('L2', 'SW', pc, addr, L2row)
        L2status = get_status(L2, L2_first_block, L2tag, L2assoc)
        if not L2status:
            evict(L2, L2_first_block, L2tag)


def get_cache_info(addr, numrows, assoc, blocksize):
    """
    Returns the row, tag, and index in the cache where the first block
    of the row is located.
    Parameters:
        addr: int = memory address being accessed
        numrows: int = number of rows in the cache
        assoc: int = associativity of the cache
        blocksize: int = blocksize of the cache
    """
    blockid = addr//blocksize
    row = blockid % numrows
    tag = blockid // numrows
    first_block = row * assoc
    return row, tag, first_block


def get_status(cache, first_block, tag, assoc):
    """
    Checks all blocks in row for the tag. If the tag is found, move the
    index of that block to the end of the LRU list, and return True.
    Otherwise, return False.
    Parameters:
        cache: list = cache structure (either L1 or L2)
        first_block: int = index of cache that holds the first block
                           of the desired row. This block contains the
                           LRU list
        tag: int = unique identifier for memory blocks
        assoc: associativity of the cache
    """
    #check each block in the row
    for i in range(0, assoc):
        #update LRU if there is a hit
        if cache[first_block + i][1] == tag:
            LRU = cache[first_block][2]     #locate the LRU list of the row
            curr = LRU.pop(i)               #move the index of the current block
            LRU.append(curr)                #to the end of the LRU list
            return True
    return False

def evict(cache, first_block, tag):
    """
    Evicts the least recently used block in the case of a miss.
    Parameters:
        cache: list = cache structure (either L1 or L2)
        first_block: int = index of cache that holds the first block
                           of the desired row. This block contains the
                           LRU list
        tag: int = unique identifier for memory blocks
    """
    LRU = cache[first_block][2]             #locate the LRU list of the row
    oldest = LRU.pop(0)                     #identify the oldest index (first in list)
    LRU.append(oldest)                      #update LRU
    cache[first_block + oldest][1] = tag    #replace data of least recent block


def simulate(instructions, mem, pc, regs, info):
    """
    Modifies registers, memory, or the program counter based on the
    16-bit instruction. All instructions except halt modify the
    program counter and recursively calls the function with this
    new pc. When the instruction is halt, the end pc is returned.
    Parameters:
        instructions: list = list of instructions as ints
        mem: list = list of memory cells with int values
        pc: int = program counter
        regs: list = list of regs
    """
    num = instructions[pc]
    opcode = num >> 13      #opcode
    a = (num >> 10) & 7     #regA
    b = (num >> 7) & 7      #regB
    c = (num >> 4) & 7      #regC
    imm = num & 127         #7-bit imm
    if (imm & 64) == 64:    #signed imm if first bit is 1
        imm -= 128
    long_imm = num & 8191   #13-bit imm

    if opcode == 0:
        if num & 15 == 8: #jr
            return simulate(instructions, mem, unsigned((regs[a] & 8191)), regs, info)
        else:
            # & (constants[2] - 1) makes sure the updated reg is 16-bit
            # the following instructions increase pc by 1
            if num & 15 == 0 and c != 0: #add (dst != $0)
                regs[c] = unsigned(regs[a] + regs[b])
            if num & 15 == 1 and c != 0: #sub (dst != $0)
                regs[c] = unsigned(regs[a] - regs[b])
            if num & 15 == 2: #or
                regs[c] = unsigned(regs[a] | regs[b])
            if num & 15 == 3: #and
                regs[c] = unsigned(regs[a] & regs[b])
            if num & 15 == 4: #slt
                if regs[a] < regs[b]: #reg values are already 16-bit unsigned
                    regs[c] = unsigned(1)
                else:
                    regs[c] = unsigned(0)
            return simulate(instructions, mem, unsigned((pc + 1) & 8191), regs, info)

    elif opcode == 1: #addi (dst != $0)
        if (b != 0):
            regs[b] = unsigned(regs[a] + imm)
        return simulate(instructions, mem, unsigned((pc + 1) & 8191), regs, info)

    elif opcode == 2: #j
        if long_imm == pc: #halt
            return pc #end of program
        else: # set pc to 13-bit unsigned
            return simulate(instructions, mem, unsigned((long_imm) & 8191), regs, info)

    elif opcode == 3: #jal
        regs[7] = (pc + 1) & (constants[2] - 1)
        return simulate(instructions, mem, unsigned((long_imm) & 8191), regs, info)

    elif opcode == 4 and b != 0: #lw
        regs[b] = unsigned(mem[(imm + regs[a]) & 8191])
        execute_lw((imm + regs[a]) & 8191, pc, info)
        return simulate(instructions, mem, unsigned((pc + 1) & 8191), regs, info)

    elif opcode == 5: #sw
        mem[(imm + regs[a]) & 8191] = regs[b]
        execute_sw((imm + regs[a]) & 8191, pc, info)
        return simulate(instructions, mem, unsigned((pc + 1) & 8191), regs, info)

    elif opcode == 6: #jeq
        if regs[a] == regs[b]:
            #only 13 most significant bits
            return simulate(instructions, mem, unsigned((pc + 1 + imm) & 8191), regs, info)
        else:
            return simulate(instructions, mem, unsigned((pc + 1) & 8191), regs, info)

    elif opcode == 7: #slti
        if regs[a] < unsigned(imm):
            regs[b] = unsigned(1)
        else:
            regs[b] = unsigned(0)
        return simulate(instructions, mem, unsigned((pc + 1) & 8191), regs, info)

def unsigned(val):
    """
    Converts val to 16-bit unsigned.
    Parameters:
        val: int = number to be stored in pc, mem, or reg
    """
    return val & (constants[2] - 1)

def parse_line(line, instructions):
    """
    Removes comments and 'ram[0] = 16'b' from each line
    and adds it to the instructions list.
    Parameters:
        line: str = line of machine code
        instructions: list = list of instructions as ints
    """
    if len(line) > 0:
        line = line.split(";",1)[0].strip()
        line = line.split("b",1)[1].strip()
        instructions.append(int(line,2))


def load_machine_code(machine_code, mem, instructions):
    """
    Loads an E20 machine code file into the list
    provided by mem. We assume that mem is
    large enough to hold the values in the machine
    code file.
    sig: list(str) -> list(int) -> NoneType
    """
    machine_code_re = re.compile("^ram\[(\d+)\] = 16'b(\d+);.*$")
    expectedaddr = 0
    for line in machine_code:
        parse_line(line, instructions)
        match = machine_code_re.match(line)
        if not match:
            raise ValueError("Can't parse line: %s" % line)
        addr, instr = match.groups()
        addr = int(addr,10)
        instr = int(instr,2)
        if addr != expectedaddr:
            raise ValueError("Memory addresses encountered out of sequence: %s" % addr)
        if addr >= len(mem):
            raise ValueError("Program too big for memory")
        expectedaddr += 1
        mem[addr] = instr

def print_cache_config(cache_name, size, assoc, blocksize, num_rows):
    """
    Prints out the correctly-formatted configuration of a cache.

    cache_name -- The name of the cache. "L1" or "L2"

    size -- The total size of the cache, measured in memory cells.
        Excludes metadata

    assoc -- The associativity of the cache. One of [1,2,4,8,16]

    blocksize -- The blocksize of the cache. One of [1,2,4,8,16,32,64])

    num_rows -- The number of rows in the given cache.

    sig: str, int, int, int, int -> NoneType
    """

    summary = "Cache %s has size %s, associativity %s, " \
        "blocksize %s, rows %s" % (cache_name,
        size, assoc, blocksize, num_rows)
    print(summary)

def print_log_entry(cache_name, status, pc, addr, row):
    """
    Prints out a correctly-formatted log entry.

    cache_name -- The name of the cache where the event
        occurred. "L1" or "L2"

    status -- The kind of cache event. "SW", "HIT", or
        "MISS"

    pc -- The program counter of the memory
        access instruction

    addr -- The memory address being accessed.

    row -- The cache row or set number where the data
        is stored.

    sig: str, str, int, int, int -> NoneType
    """
    log_entry = "{event:8s} pc:{pc:5d}\taddr:{addr:5d}\t" \
        "row:{row:4d}".format(row=row, pc=pc, addr=addr,
            event = cache_name + " " + status)
    print(log_entry)

def main():
    parser = argparse.ArgumentParser(description='Simulate E20 cache')
    parser.add_argument('filename', help=
        'The file containing machine code, typically with .bin suffix')
    parser.add_argument('--cache', help=
        'Cache configuration: size,associativity,blocksize (for one cache) '
        'or size,associativity,blocksize,size,associativity,blocksize (for two caches)')
    cmdline = parser.parse_args()


    mem = [0] * constants[1]    # 2^13 memory slots
    regs = [0] * constants[0]   # 8 registers
    instructions = []

    with open(cmdline.filename) as file:
    #Load file and parse using load_machine_code
        load_machine_code(file, mem, instructions)

    if cmdline.cache is not None:
        parts = cmdline.cache.split(",")
        if len(parts) == 3:
            [L1size, L1assoc, L1blocksize] = [int(x) for x in parts]
            L1numrows = (L1size // L1blocksize) // L1assoc #calculate number of rows
            print_cache_config('L1', L1size, L1assoc, L1blocksize, L1numrows)
            L1 = create_cache(L1numrows, L1assoc) #create L1
            info = [L1, L1numrows, L1assoc, L1blocksize]
            simulate(instructions, mem, 0, regs, info)  #execute E20 program with cache info

        elif len(parts) == 6:
            [L1size, L1assoc, L1blocksize, L2size, L2assoc, L2blocksize] = [int(x) for x in parts]
            L1numrows = (L1size // L1blocksize) // L1assoc #calculate number of rows in L1
            L2numrows = (L2size // L2blocksize) // L2assoc #calculate number of rows in L2
            print_cache_config('L1', L1size, L1assoc, L1blocksize, L1numrows)
            print_cache_config('L2', L2size, L2assoc, L2blocksize, L2numrows)
            L1 = create_cache(L1numrows, L1assoc)   #create L1
            L2 = create_cache(L2numrows, L2assoc)   #create L2
            info = [L1, L1numrows, L1assoc, L1blocksize, L2, L2numrows, L2assoc, L2blocksize]
            simulate(instructions, mem, 0, regs, info)  #execute E20 program with cache info

        else:
            raise Exception("Invalid cache config")



if __name__ == "__main__":
    main()
#ra0Eequ6ucie6Jei0koh6phishohm9
