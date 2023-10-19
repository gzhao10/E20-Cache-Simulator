# We're testing that write-through works.
# A write should cache the value,
# even if it is already in the cache.

movi $1, 42
lw $2, 50($0) # this should always be a miss
sw $1, 50($0)


lw $2, 50($0) # this should always be a hit

halt
#--
#--
#--MACHINE CODE
# ram[0] = 16'b0010000010101010;		// movi $1,42
# ram[1] = 16'b1000000100110010;		// lw $2,50($0)
# ram[2] = 16'b1010000010110010;		// sw $1,50($0)
# ram[3] = 16'b1000000100110010;		// lw $2,50($0)
# ram[4] = 16'b0100000000000100;		// halt 
#--
#--
#--EXECUTION OUTPUT
# write-through.bin --cache 4,1,1
# 	Cache L1 has size 4, associativity 1, blocksize 1, rows 4
# 	L1 MISS  pc:    1	addr:   50	row:   2
# 	L1 SW    pc:    2	addr:   50	row:   2
# 	L1 HIT   pc:    3	addr:   50	row:   2
# 
# write-through.bin --cache 16,1,4
# 	Cache L1 has size 16, associativity 1, blocksize 4, rows 4
# 	L1 MISS  pc:    1	addr:   50	row:   0
# 	L1 SW    pc:    2	addr:   50	row:   0
# 	L1 HIT   pc:    3	addr:   50	row:   0
# 
# write-through.bin --cache 32,2,4
# 	Cache L1 has size 32, associativity 2, blocksize 4, rows 4
# 	L1 MISS  pc:    1	addr:   50	row:   0
# 	L1 SW    pc:    2	addr:   50	row:   0
# 	L1 HIT   pc:    3	addr:   50	row:   0
# 
# write-through.bin --cache 4,1,2,8,4,2
# 	Cache L1 has size 4, associativity 1, blocksize 2, rows 2
# 	Cache L2 has size 8, associativity 4, blocksize 2, rows 1
# 	L1 MISS  pc:    1	addr:   50	row:   1
# 	L2 MISS  pc:    1	addr:   50	row:   0
# 	L1 SW    pc:    2	addr:   50	row:   1
# 	L2 SW    pc:    2	addr:   50	row:   0
# 	L1 HIT   pc:    3	addr:   50	row:   1
# 
# write-through.bin --cache 4,1,1,64,4,4
# 	Cache L1 has size 4, associativity 1, blocksize 1, rows 4
# 	Cache L2 has size 64, associativity 4, blocksize 4, rows 4
# 	L1 MISS  pc:    1	addr:   50	row:   2
# 	L2 MISS  pc:    1	addr:   50	row:   0
# 	L1 SW    pc:    2	addr:   50	row:   2
# 	L2 SW    pc:    2	addr:   50	row:   0
# 	L1 HIT   pc:    3	addr:   50	row:   2
# 
