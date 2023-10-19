    movi $1, foo
    lw $2, 0($1)
    lw $3, 1($1)
    add $4, $2, $3
    sw $4, foo($0)
    halt
foo:
    .fill 10
    .fill 20
#--
#--
#--MACHINE CODE
# ram[0] = 16'b0010000010000110;		// movi $1,foo
# ram[1] = 16'b1000010100000000;		// lw $2,0($1)
# ram[2] = 16'b1000010110000001;		// lw $3,1($1)
# ram[3] = 16'b0000100111000000;		// add $4,$2,$3
# ram[4] = 16'b1010001000000110;		// sw $4,foo($0)
# ram[5] = 16'b0100000000000101;		// halt 
# ram[6] = 16'b0000000000001010;		// foo: .fill 10
# ram[7] = 16'b0000000000010100;		// .fill 20
#--
#--
#--EXECUTION OUTPUT
# test.bin --cache 4,1,1
# 	Cache L1 has size 4, associativity 1, blocksize 1, rows 4
# 	L1 MISS  pc:    1	addr:    6	row:   2
# 	L1 MISS  pc:    2	addr:    7	row:   3
# 	L1 SW    pc:    4	addr:    6	row:   2
# 
# test.bin --cache 16,1,4
# 	Cache L1 has size 16, associativity 1, blocksize 4, rows 4
# 	L1 MISS  pc:    1	addr:    6	row:   1
# 	L1 HIT   pc:    2	addr:    7	row:   1
# 	L1 SW    pc:    4	addr:    6	row:   1
# 
# test.bin --cache 32,2,4
# 	Cache L1 has size 32, associativity 2, blocksize 4, rows 4
# 	L1 MISS  pc:    1	addr:    6	row:   1
# 	L1 HIT   pc:    2	addr:    7	row:   1
# 	L1 SW    pc:    4	addr:    6	row:   1
# 
# test.bin --cache 4,1,2,8,4,2
# 	Cache L1 has size 4, associativity 1, blocksize 2, rows 2
# 	Cache L2 has size 8, associativity 4, blocksize 2, rows 1
# 	L1 MISS  pc:    1	addr:    6	row:   1
# 	L2 MISS  pc:    1	addr:    6	row:   0
# 	L1 HIT   pc:    2	addr:    7	row:   1
# 	L1 SW    pc:    4	addr:    6	row:   1
# 	L2 SW    pc:    4	addr:    6	row:   0
# 
# test.bin --cache 4,1,1,64,4,4
# 	Cache L1 has size 4, associativity 1, blocksize 1, rows 4
# 	Cache L2 has size 64, associativity 4, blocksize 4, rows 4
# 	L1 MISS  pc:    1	addr:    6	row:   2
# 	L2 MISS  pc:    1	addr:    6	row:   1
# 	L1 MISS  pc:    2	addr:    7	row:   3
# 	L2 HIT   pc:    2	addr:    7	row:   1
# 	L1 SW    pc:    4	addr:    6	row:   2
# 	L2 SW    pc:    4	addr:    6	row:   1
# 
