movi $1, 32
add $1, $1, $1
add $2, $1, $1
add $2, $2, $2
add $2, $2, $2
add $2, $2, $2

lw $5, 0($2)
add $3, $2, $1
lw $6, 0($3)

lw $5, 0($2)
add $3, $2, $1
lw $6, 0($3)

lw $5, 0($2)
add $3, $2, $1
lw $6, 0($3)

lw $7, 50($2) 

add $3, $2, $1
lw $6, 0($3)
lw $5, 0($2)


halt
#--
#--
#--MACHINE CODE
# ram[0] = 16'b0010000010100000;		// movi $1,32
# ram[1] = 16'b0000010010010000;		// add $1,$1,$1
# ram[2] = 16'b0000010010100000;		// add $2,$1,$1
# ram[3] = 16'b0000100100100000;		// add $2,$2,$2
# ram[4] = 16'b0000100100100000;		// add $2,$2,$2
# ram[5] = 16'b0000100100100000;		// add $2,$2,$2
# ram[6] = 16'b1000101010000000;		// lw $5,0($2)
# ram[7] = 16'b0000100010110000;		// add $3,$2,$1
# ram[8] = 16'b1000111100000000;		// lw $6,0($3)
# ram[9] = 16'b1000101010000000;		// lw $5,0($2)
# ram[10] = 16'b0000100010110000;		// add $3,$2,$1
# ram[11] = 16'b1000111100000000;		// lw $6,0($3)
# ram[12] = 16'b1000101010000000;		// lw $5,0($2)
# ram[13] = 16'b0000100010110000;		// add $3,$2,$1
# ram[14] = 16'b1000111100000000;		// lw $6,0($3)
# ram[15] = 16'b1000101110110010;		// lw $7,50($2)
# ram[16] = 16'b0000100010110000;		// add $3,$2,$1
# ram[17] = 16'b1000111100000000;		// lw $6,0($3)
# ram[18] = 16'b1000101010000000;		// lw $5,0($2)
# ram[19] = 16'b0100000000010011;		// halt 
#--
#--
#--EXECUTION OUTPUT
# assoc2.bin --cache 4,1,1
# 	Cache L1 has size 4, associativity 1, blocksize 1, rows 4
# 	L1 MISS  pc:    6	addr: 1024	row:   0
# 	L1 MISS  pc:    8	addr: 1088	row:   0
# 	L1 MISS  pc:    9	addr: 1024	row:   0
# 	L1 MISS  pc:   11	addr: 1088	row:   0
# 	L1 MISS  pc:   12	addr: 1024	row:   0
# 	L1 MISS  pc:   14	addr: 1088	row:   0
# 	L1 MISS  pc:   15	addr: 1074	row:   2
# 	L1 HIT   pc:   17	addr: 1088	row:   0
# 	L1 MISS  pc:   18	addr: 1024	row:   0
# 
# assoc2.bin --cache 16,1,4
# 	Cache L1 has size 16, associativity 1, blocksize 4, rows 4
# 	L1 MISS  pc:    6	addr: 1024	row:   0
# 	L1 MISS  pc:    8	addr: 1088	row:   0
# 	L1 MISS  pc:    9	addr: 1024	row:   0
# 	L1 MISS  pc:   11	addr: 1088	row:   0
# 	L1 MISS  pc:   12	addr: 1024	row:   0
# 	L1 MISS  pc:   14	addr: 1088	row:   0
# 	L1 MISS  pc:   15	addr: 1074	row:   0
# 	L1 MISS  pc:   17	addr: 1088	row:   0
# 	L1 MISS  pc:   18	addr: 1024	row:   0
# 
# assoc2.bin --cache 32,2,4
# 	Cache L1 has size 32, associativity 2, blocksize 4, rows 4
# 	L1 MISS  pc:    6	addr: 1024	row:   0
# 	L1 MISS  pc:    8	addr: 1088	row:   0
# 	L1 HIT   pc:    9	addr: 1024	row:   0
# 	L1 HIT   pc:   11	addr: 1088	row:   0
# 	L1 HIT   pc:   12	addr: 1024	row:   0
# 	L1 HIT   pc:   14	addr: 1088	row:   0
# 	L1 MISS  pc:   15	addr: 1074	row:   0
# 	L1 HIT   pc:   17	addr: 1088	row:   0
# 	L1 MISS  pc:   18	addr: 1024	row:   0
# 
# assoc2.bin --cache 4,1,2,8,4,2
# 	Cache L1 has size 4, associativity 1, blocksize 2, rows 2
# 	Cache L2 has size 8, associativity 4, blocksize 2, rows 1
# 	L1 MISS  pc:    6	addr: 1024	row:   0
# 	L2 MISS  pc:    6	addr: 1024	row:   0
# 	L1 MISS  pc:    8	addr: 1088	row:   0
# 	L2 MISS  pc:    8	addr: 1088	row:   0
# 	L1 MISS  pc:    9	addr: 1024	row:   0
# 	L2 HIT   pc:    9	addr: 1024	row:   0
# 	L1 MISS  pc:   11	addr: 1088	row:   0
# 	L2 HIT   pc:   11	addr: 1088	row:   0
# 	L1 MISS  pc:   12	addr: 1024	row:   0
# 	L2 HIT   pc:   12	addr: 1024	row:   0
# 	L1 MISS  pc:   14	addr: 1088	row:   0
# 	L2 HIT   pc:   14	addr: 1088	row:   0
# 	L1 MISS  pc:   15	addr: 1074	row:   1
# 	L2 MISS  pc:   15	addr: 1074	row:   0
# 	L1 HIT   pc:   17	addr: 1088	row:   0
# 	L1 MISS  pc:   18	addr: 1024	row:   0
# 	L2 HIT   pc:   18	addr: 1024	row:   0
# 
# assoc2.bin --cache 4,1,1,64,4,4
# 	Cache L1 has size 4, associativity 1, blocksize 1, rows 4
# 	Cache L2 has size 64, associativity 4, blocksize 4, rows 4
# 	L1 MISS  pc:    6	addr: 1024	row:   0
# 	L2 MISS  pc:    6	addr: 1024	row:   0
# 	L1 MISS  pc:    8	addr: 1088	row:   0
# 	L2 MISS  pc:    8	addr: 1088	row:   0
# 	L1 MISS  pc:    9	addr: 1024	row:   0
# 	L2 HIT   pc:    9	addr: 1024	row:   0
# 	L1 MISS  pc:   11	addr: 1088	row:   0
# 	L2 HIT   pc:   11	addr: 1088	row:   0
# 	L1 MISS  pc:   12	addr: 1024	row:   0
# 	L2 HIT   pc:   12	addr: 1024	row:   0
# 	L1 MISS  pc:   14	addr: 1088	row:   0
# 	L2 HIT   pc:   14	addr: 1088	row:   0
# 	L1 MISS  pc:   15	addr: 1074	row:   2
# 	L2 MISS  pc:   15	addr: 1074	row:   0
# 	L1 HIT   pc:   17	addr: 1088	row:   0
# 	L1 MISS  pc:   18	addr: 1024	row:   0
# 	L2 HIT   pc:   18	addr: 1024	row:   0
# 
