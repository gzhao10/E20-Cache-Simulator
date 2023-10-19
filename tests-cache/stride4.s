movi $1, 53
lw $7, hundred($0)

sw $1, 0($7)
sw $1, 4($7)
sw $1, 8($7)
sw $1, 12($7)
sw $1, 16($7)

movi $2, 10
loop:
jeq $2, $0, done
addi $2, $2, -1

lw $1, 0($7)
lw $1, 4($7)
lw $1, 8($7)
lw $1, 12($7)
lw $1, 16($7)


j loop

done: halt
hundred: .fill 100
#--
#--
#--MACHINE CODE
# ram[0] = 16'b0010000010110101;		// movi $1,53
# ram[1] = 16'b1000001110010001;		// lw $7,hundred($0)
# ram[2] = 16'b1011110010000000;		// sw $1,0($7)
# ram[3] = 16'b1011110010000100;		// sw $1,4($7)
# ram[4] = 16'b1011110010001000;		// sw $1,8($7)
# ram[5] = 16'b1011110010001100;		// sw $1,12($7)
# ram[6] = 16'b1011110010010000;		// sw $1,16($7)
# ram[7] = 16'b0010000100001010;		// movi $2,10
# ram[8] = 16'b1100100000000111;		// loop: jeq $2,$0,done
# ram[9] = 16'b0010100101111111;		// addi $2,$2,-1
# ram[10] = 16'b1001110010000000;		// lw $1,0($7)
# ram[11] = 16'b1001110010000100;		// lw $1,4($7)
# ram[12] = 16'b1001110010001000;		// lw $1,8($7)
# ram[13] = 16'b1001110010001100;		// lw $1,12($7)
# ram[14] = 16'b1001110010010000;		// lw $1,16($7)
# ram[15] = 16'b0100000000001000;		// j loop
# ram[16] = 16'b0100000000010000;		// done: halt 
# ram[17] = 16'b0000000001100100;		// hundred: .fill 100
#--
#--
#--EXECUTION OUTPUT
# stride4.bin --cache 4,1,1
# 	Cache L1 has size 4, associativity 1, blocksize 1, rows 4
# 	L1 MISS  pc:    1	addr:   17	row:   1
# 	L1 SW    pc:    2	addr:  100	row:   0
# 	L1 SW    pc:    3	addr:  104	row:   0
# 	L1 SW    pc:    4	addr:  108	row:   0
# 	L1 SW    pc:    5	addr:  112	row:   0
# 	L1 SW    pc:    6	addr:  116	row:   0
# 	L1 MISS  pc:   10	addr:  100	row:   0
# 	L1 MISS  pc:   11	addr:  104	row:   0
# 	L1 MISS  pc:   12	addr:  108	row:   0
# 	L1 MISS  pc:   13	addr:  112	row:   0
# 	L1 MISS  pc:   14	addr:  116	row:   0
# 	L1 MISS  pc:   10	addr:  100	row:   0
# 	L1 MISS  pc:   11	addr:  104	row:   0
# 	L1 MISS  pc:   12	addr:  108	row:   0
# 	L1 MISS  pc:   13	addr:  112	row:   0
# 	L1 MISS  pc:   14	addr:  116	row:   0
# 	L1 MISS  pc:   10	addr:  100	row:   0
# 	L1 MISS  pc:   11	addr:  104	row:   0
# 	L1 MISS  pc:   12	addr:  108	row:   0
# 	L1 MISS  pc:   13	addr:  112	row:   0
# 	L1 MISS  pc:   14	addr:  116	row:   0
# 	L1 MISS  pc:   10	addr:  100	row:   0
# 	L1 MISS  pc:   11	addr:  104	row:   0
# 	L1 MISS  pc:   12	addr:  108	row:   0
# 	L1 MISS  pc:   13	addr:  112	row:   0
# 	L1 MISS  pc:   14	addr:  116	row:   0
# 	L1 MISS  pc:   10	addr:  100	row:   0
# 	L1 MISS  pc:   11	addr:  104	row:   0
# 	L1 MISS  pc:   12	addr:  108	row:   0
# 	L1 MISS  pc:   13	addr:  112	row:   0
# 	L1 MISS  pc:   14	addr:  116	row:   0
# 	L1 MISS  pc:   10	addr:  100	row:   0
# 	L1 MISS  pc:   11	addr:  104	row:   0
# 	L1 MISS  pc:   12	addr:  108	row:   0
# 	L1 MISS  pc:   13	addr:  112	row:   0
# 	L1 MISS  pc:   14	addr:  116	row:   0
# 	L1 MISS  pc:   10	addr:  100	row:   0
# 	L1 MISS  pc:   11	addr:  104	row:   0
# 	L1 MISS  pc:   12	addr:  108	row:   0
# 	L1 MISS  pc:   13	addr:  112	row:   0
# 	L1 MISS  pc:   14	addr:  116	row:   0
# 	L1 MISS  pc:   10	addr:  100	row:   0
# 	L1 MISS  pc:   11	addr:  104	row:   0
# 	L1 MISS  pc:   12	addr:  108	row:   0
# 	L1 MISS  pc:   13	addr:  112	row:   0
# 	L1 MISS  pc:   14	addr:  116	row:   0
# 	L1 MISS  pc:   10	addr:  100	row:   0
# 	L1 MISS  pc:   11	addr:  104	row:   0
# 	L1 MISS  pc:   12	addr:  108	row:   0
# 	L1 MISS  pc:   13	addr:  112	row:   0
# 	L1 MISS  pc:   14	addr:  116	row:   0
# 	L1 MISS  pc:   10	addr:  100	row:   0
# 	L1 MISS  pc:   11	addr:  104	row:   0
# 	L1 MISS  pc:   12	addr:  108	row:   0
# 	L1 MISS  pc:   13	addr:  112	row:   0
# 	L1 MISS  pc:   14	addr:  116	row:   0
# 
# stride4.bin --cache 16,1,4
# 	Cache L1 has size 16, associativity 1, blocksize 4, rows 4
# 	L1 MISS  pc:    1	addr:   17	row:   0
# 	L1 SW    pc:    2	addr:  100	row:   1
# 	L1 SW    pc:    3	addr:  104	row:   2
# 	L1 SW    pc:    4	addr:  108	row:   3
# 	L1 SW    pc:    5	addr:  112	row:   0
# 	L1 SW    pc:    6	addr:  116	row:   1
# 	L1 MISS  pc:   10	addr:  100	row:   1
# 	L1 HIT   pc:   11	addr:  104	row:   2
# 	L1 HIT   pc:   12	addr:  108	row:   3
# 	L1 HIT   pc:   13	addr:  112	row:   0
# 	L1 MISS  pc:   14	addr:  116	row:   1
# 	L1 MISS  pc:   10	addr:  100	row:   1
# 	L1 HIT   pc:   11	addr:  104	row:   2
# 	L1 HIT   pc:   12	addr:  108	row:   3
# 	L1 HIT   pc:   13	addr:  112	row:   0
# 	L1 MISS  pc:   14	addr:  116	row:   1
# 	L1 MISS  pc:   10	addr:  100	row:   1
# 	L1 HIT   pc:   11	addr:  104	row:   2
# 	L1 HIT   pc:   12	addr:  108	row:   3
# 	L1 HIT   pc:   13	addr:  112	row:   0
# 	L1 MISS  pc:   14	addr:  116	row:   1
# 	L1 MISS  pc:   10	addr:  100	row:   1
# 	L1 HIT   pc:   11	addr:  104	row:   2
# 	L1 HIT   pc:   12	addr:  108	row:   3
# 	L1 HIT   pc:   13	addr:  112	row:   0
# 	L1 MISS  pc:   14	addr:  116	row:   1
# 	L1 MISS  pc:   10	addr:  100	row:   1
# 	L1 HIT   pc:   11	addr:  104	row:   2
# 	L1 HIT   pc:   12	addr:  108	row:   3
# 	L1 HIT   pc:   13	addr:  112	row:   0
# 	L1 MISS  pc:   14	addr:  116	row:   1
# 	L1 MISS  pc:   10	addr:  100	row:   1
# 	L1 HIT   pc:   11	addr:  104	row:   2
# 	L1 HIT   pc:   12	addr:  108	row:   3
# 	L1 HIT   pc:   13	addr:  112	row:   0
# 	L1 MISS  pc:   14	addr:  116	row:   1
# 	L1 MISS  pc:   10	addr:  100	row:   1
# 	L1 HIT   pc:   11	addr:  104	row:   2
# 	L1 HIT   pc:   12	addr:  108	row:   3
# 	L1 HIT   pc:   13	addr:  112	row:   0
# 	L1 MISS  pc:   14	addr:  116	row:   1
# 	L1 MISS  pc:   10	addr:  100	row:   1
# 	L1 HIT   pc:   11	addr:  104	row:   2
# 	L1 HIT   pc:   12	addr:  108	row:   3
# 	L1 HIT   pc:   13	addr:  112	row:   0
# 	L1 MISS  pc:   14	addr:  116	row:   1
# 	L1 MISS  pc:   10	addr:  100	row:   1
# 	L1 HIT   pc:   11	addr:  104	row:   2
# 	L1 HIT   pc:   12	addr:  108	row:   3
# 	L1 HIT   pc:   13	addr:  112	row:   0
# 	L1 MISS  pc:   14	addr:  116	row:   1
# 	L1 MISS  pc:   10	addr:  100	row:   1
# 	L1 HIT   pc:   11	addr:  104	row:   2
# 	L1 HIT   pc:   12	addr:  108	row:   3
# 	L1 HIT   pc:   13	addr:  112	row:   0
# 	L1 MISS  pc:   14	addr:  116	row:   1
# 
# stride4.bin --cache 32,2,4
# 	Cache L1 has size 32, associativity 2, blocksize 4, rows 4
# 	L1 MISS  pc:    1	addr:   17	row:   0
# 	L1 SW    pc:    2	addr:  100	row:   1
# 	L1 SW    pc:    3	addr:  104	row:   2
# 	L1 SW    pc:    4	addr:  108	row:   3
# 	L1 SW    pc:    5	addr:  112	row:   0
# 	L1 SW    pc:    6	addr:  116	row:   1
# 	L1 HIT   pc:   10	addr:  100	row:   1
# 	L1 HIT   pc:   11	addr:  104	row:   2
# 	L1 HIT   pc:   12	addr:  108	row:   3
# 	L1 HIT   pc:   13	addr:  112	row:   0
# 	L1 HIT   pc:   14	addr:  116	row:   1
# 	L1 HIT   pc:   10	addr:  100	row:   1
# 	L1 HIT   pc:   11	addr:  104	row:   2
# 	L1 HIT   pc:   12	addr:  108	row:   3
# 	L1 HIT   pc:   13	addr:  112	row:   0
# 	L1 HIT   pc:   14	addr:  116	row:   1
# 	L1 HIT   pc:   10	addr:  100	row:   1
# 	L1 HIT   pc:   11	addr:  104	row:   2
# 	L1 HIT   pc:   12	addr:  108	row:   3
# 	L1 HIT   pc:   13	addr:  112	row:   0
# 	L1 HIT   pc:   14	addr:  116	row:   1
# 	L1 HIT   pc:   10	addr:  100	row:   1
# 	L1 HIT   pc:   11	addr:  104	row:   2
# 	L1 HIT   pc:   12	addr:  108	row:   3
# 	L1 HIT   pc:   13	addr:  112	row:   0
# 	L1 HIT   pc:   14	addr:  116	row:   1
# 	L1 HIT   pc:   10	addr:  100	row:   1
# 	L1 HIT   pc:   11	addr:  104	row:   2
# 	L1 HIT   pc:   12	addr:  108	row:   3
# 	L1 HIT   pc:   13	addr:  112	row:   0
# 	L1 HIT   pc:   14	addr:  116	row:   1
# 	L1 HIT   pc:   10	addr:  100	row:   1
# 	L1 HIT   pc:   11	addr:  104	row:   2
# 	L1 HIT   pc:   12	addr:  108	row:   3
# 	L1 HIT   pc:   13	addr:  112	row:   0
# 	L1 HIT   pc:   14	addr:  116	row:   1
# 	L1 HIT   pc:   10	addr:  100	row:   1
# 	L1 HIT   pc:   11	addr:  104	row:   2
# 	L1 HIT   pc:   12	addr:  108	row:   3
# 	L1 HIT   pc:   13	addr:  112	row:   0
# 	L1 HIT   pc:   14	addr:  116	row:   1
# 	L1 HIT   pc:   10	addr:  100	row:   1
# 	L1 HIT   pc:   11	addr:  104	row:   2
# 	L1 HIT   pc:   12	addr:  108	row:   3
# 	L1 HIT   pc:   13	addr:  112	row:   0
# 	L1 HIT   pc:   14	addr:  116	row:   1
# 	L1 HIT   pc:   10	addr:  100	row:   1
# 	L1 HIT   pc:   11	addr:  104	row:   2
# 	L1 HIT   pc:   12	addr:  108	row:   3
# 	L1 HIT   pc:   13	addr:  112	row:   0
# 	L1 HIT   pc:   14	addr:  116	row:   1
# 	L1 HIT   pc:   10	addr:  100	row:   1
# 	L1 HIT   pc:   11	addr:  104	row:   2
# 	L1 HIT   pc:   12	addr:  108	row:   3
# 	L1 HIT   pc:   13	addr:  112	row:   0
# 	L1 HIT   pc:   14	addr:  116	row:   1
# 
# stride4.bin --cache 4,1,2,8,4,2
# 	Cache L1 has size 4, associativity 1, blocksize 2, rows 2
# 	Cache L2 has size 8, associativity 4, blocksize 2, rows 1
# 	L1 MISS  pc:    1	addr:   17	row:   0
# 	L2 MISS  pc:    1	addr:   17	row:   0
# 	L1 SW    pc:    2	addr:  100	row:   0
# 	L2 SW    pc:    2	addr:  100	row:   0
# 	L1 SW    pc:    3	addr:  104	row:   0
# 	L2 SW    pc:    3	addr:  104	row:   0
# 	L1 SW    pc:    4	addr:  108	row:   0
# 	L2 SW    pc:    4	addr:  108	row:   0
# 	L1 SW    pc:    5	addr:  112	row:   0
# 	L2 SW    pc:    5	addr:  112	row:   0
# 	L1 SW    pc:    6	addr:  116	row:   0
# 	L2 SW    pc:    6	addr:  116	row:   0
# 	L1 MISS  pc:   10	addr:  100	row:   0
# 	L2 MISS  pc:   10	addr:  100	row:   0
# 	L1 MISS  pc:   11	addr:  104	row:   0
# 	L2 MISS  pc:   11	addr:  104	row:   0
# 	L1 MISS  pc:   12	addr:  108	row:   0
# 	L2 MISS  pc:   12	addr:  108	row:   0
# 	L1 MISS  pc:   13	addr:  112	row:   0
# 	L2 MISS  pc:   13	addr:  112	row:   0
# 	L1 MISS  pc:   14	addr:  116	row:   0
# 	L2 MISS  pc:   14	addr:  116	row:   0
# 	L1 MISS  pc:   10	addr:  100	row:   0
# 	L2 MISS  pc:   10	addr:  100	row:   0
# 	L1 MISS  pc:   11	addr:  104	row:   0
# 	L2 MISS  pc:   11	addr:  104	row:   0
# 	L1 MISS  pc:   12	addr:  108	row:   0
# 	L2 MISS  pc:   12	addr:  108	row:   0
# 	L1 MISS  pc:   13	addr:  112	row:   0
# 	L2 MISS  pc:   13	addr:  112	row:   0
# 	L1 MISS  pc:   14	addr:  116	row:   0
# 	L2 MISS  pc:   14	addr:  116	row:   0
# 	L1 MISS  pc:   10	addr:  100	row:   0
# 	L2 MISS  pc:   10	addr:  100	row:   0
# 	L1 MISS  pc:   11	addr:  104	row:   0
# 	L2 MISS  pc:   11	addr:  104	row:   0
# 	L1 MISS  pc:   12	addr:  108	row:   0
# 	L2 MISS  pc:   12	addr:  108	row:   0
# 	L1 MISS  pc:   13	addr:  112	row:   0
# 	L2 MISS  pc:   13	addr:  112	row:   0
# 	L1 MISS  pc:   14	addr:  116	row:   0
# 	L2 MISS  pc:   14	addr:  116	row:   0
# 	L1 MISS  pc:   10	addr:  100	row:   0
# 	L2 MISS  pc:   10	addr:  100	row:   0
# 	L1 MISS  pc:   11	addr:  104	row:   0
# 	L2 MISS  pc:   11	addr:  104	row:   0
# 	L1 MISS  pc:   12	addr:  108	row:   0
# 	L2 MISS  pc:   12	addr:  108	row:   0
# 	L1 MISS  pc:   13	addr:  112	row:   0
# 	L2 MISS  pc:   13	addr:  112	row:   0
# 	L1 MISS  pc:   14	addr:  116	row:   0
# 	L2 MISS  pc:   14	addr:  116	row:   0
# 	L1 MISS  pc:   10	addr:  100	row:   0
# 	L2 MISS  pc:   10	addr:  100	row:   0
# 	L1 MISS  pc:   11	addr:  104	row:   0
# 	L2 MISS  pc:   11	addr:  104	row:   0
# 	L1 MISS  pc:   12	addr:  108	row:   0
# 	L2 MISS  pc:   12	addr:  108	row:   0
# 	L1 MISS  pc:   13	addr:  112	row:   0
# 	L2 MISS  pc:   13	addr:  112	row:   0
# 	L1 MISS  pc:   14	addr:  116	row:   0
# 	L2 MISS  pc:   14	addr:  116	row:   0
# 	L1 MISS  pc:   10	addr:  100	row:   0
# 	L2 MISS  pc:   10	addr:  100	row:   0
# 	L1 MISS  pc:   11	addr:  104	row:   0
# 	L2 MISS  pc:   11	addr:  104	row:   0
# 	L1 MISS  pc:   12	addr:  108	row:   0
# 	L2 MISS  pc:   12	addr:  108	row:   0
# 	L1 MISS  pc:   13	addr:  112	row:   0
# 	L2 MISS  pc:   13	addr:  112	row:   0
# 	L1 MISS  pc:   14	addr:  116	row:   0
# 	L2 MISS  pc:   14	addr:  116	row:   0
# 	L1 MISS  pc:   10	addr:  100	row:   0
# 	L2 MISS  pc:   10	addr:  100	row:   0
# 	L1 MISS  pc:   11	addr:  104	row:   0
# 	L2 MISS  pc:   11	addr:  104	row:   0
# 	L1 MISS  pc:   12	addr:  108	row:   0
# 	L2 MISS  pc:   12	addr:  108	row:   0
# 	L1 MISS  pc:   13	addr:  112	row:   0
# 	L2 MISS  pc:   13	addr:  112	row:   0
# 	L1 MISS  pc:   14	addr:  116	row:   0
# 	L2 MISS  pc:   14	addr:  116	row:   0
# 	L1 MISS  pc:   10	addr:  100	row:   0
# 	L2 MISS  pc:   10	addr:  100	row:   0
# 	L1 MISS  pc:   11	addr:  104	row:   0
# 	L2 MISS  pc:   11	addr:  104	row:   0
# 	L1 MISS  pc:   12	addr:  108	row:   0
# 	L2 MISS  pc:   12	addr:  108	row:   0
# 	L1 MISS  pc:   13	addr:  112	row:   0
# 	L2 MISS  pc:   13	addr:  112	row:   0
# 	L1 MISS  pc:   14	addr:  116	row:   0
# 	L2 MISS  pc:   14	addr:  116	row:   0
# 	L1 MISS  pc:   10	addr:  100	row:   0
# 	L2 MISS  pc:   10	addr:  100	row:   0
# 	L1 MISS  pc:   11	addr:  104	row:   0
# 	L2 MISS  pc:   11	addr:  104	row:   0
# 	L1 MISS  pc:   12	addr:  108	row:   0
# 	L2 MISS  pc:   12	addr:  108	row:   0
# 	L1 MISS  pc:   13	addr:  112	row:   0
# 	L2 MISS  pc:   13	addr:  112	row:   0
# 	L1 MISS  pc:   14	addr:  116	row:   0
# 	L2 MISS  pc:   14	addr:  116	row:   0
# 	L1 MISS  pc:   10	addr:  100	row:   0
# 	L2 MISS  pc:   10	addr:  100	row:   0
# 	L1 MISS  pc:   11	addr:  104	row:   0
# 	L2 MISS  pc:   11	addr:  104	row:   0
# 	L1 MISS  pc:   12	addr:  108	row:   0
# 	L2 MISS  pc:   12	addr:  108	row:   0
# 	L1 MISS  pc:   13	addr:  112	row:   0
# 	L2 MISS  pc:   13	addr:  112	row:   0
# 	L1 MISS  pc:   14	addr:  116	row:   0
# 	L2 MISS  pc:   14	addr:  116	row:   0
# 
# stride4.bin --cache 4,1,1,64,4,4
# 	Cache L1 has size 4, associativity 1, blocksize 1, rows 4
# 	Cache L2 has size 64, associativity 4, blocksize 4, rows 4
# 	L1 MISS  pc:    1	addr:   17	row:   1
# 	L2 MISS  pc:    1	addr:   17	row:   0
# 	L1 SW    pc:    2	addr:  100	row:   0
# 	L2 SW    pc:    2	addr:  100	row:   1
# 	L1 SW    pc:    3	addr:  104	row:   0
# 	L2 SW    pc:    3	addr:  104	row:   2
# 	L1 SW    pc:    4	addr:  108	row:   0
# 	L2 SW    pc:    4	addr:  108	row:   3
# 	L1 SW    pc:    5	addr:  112	row:   0
# 	L2 SW    pc:    5	addr:  112	row:   0
# 	L1 SW    pc:    6	addr:  116	row:   0
# 	L2 SW    pc:    6	addr:  116	row:   1
# 	L1 MISS  pc:   10	addr:  100	row:   0
# 	L2 HIT   pc:   10	addr:  100	row:   1
# 	L1 MISS  pc:   11	addr:  104	row:   0
# 	L2 HIT   pc:   11	addr:  104	row:   2
# 	L1 MISS  pc:   12	addr:  108	row:   0
# 	L2 HIT   pc:   12	addr:  108	row:   3
# 	L1 MISS  pc:   13	addr:  112	row:   0
# 	L2 HIT   pc:   13	addr:  112	row:   0
# 	L1 MISS  pc:   14	addr:  116	row:   0
# 	L2 HIT   pc:   14	addr:  116	row:   1
# 	L1 MISS  pc:   10	addr:  100	row:   0
# 	L2 HIT   pc:   10	addr:  100	row:   1
# 	L1 MISS  pc:   11	addr:  104	row:   0
# 	L2 HIT   pc:   11	addr:  104	row:   2
# 	L1 MISS  pc:   12	addr:  108	row:   0
# 	L2 HIT   pc:   12	addr:  108	row:   3
# 	L1 MISS  pc:   13	addr:  112	row:   0
# 	L2 HIT   pc:   13	addr:  112	row:   0
# 	L1 MISS  pc:   14	addr:  116	row:   0
# 	L2 HIT   pc:   14	addr:  116	row:   1
# 	L1 MISS  pc:   10	addr:  100	row:   0
# 	L2 HIT   pc:   10	addr:  100	row:   1
# 	L1 MISS  pc:   11	addr:  104	row:   0
# 	L2 HIT   pc:   11	addr:  104	row:   2
# 	L1 MISS  pc:   12	addr:  108	row:   0
# 	L2 HIT   pc:   12	addr:  108	row:   3
# 	L1 MISS  pc:   13	addr:  112	row:   0
# 	L2 HIT   pc:   13	addr:  112	row:   0
# 	L1 MISS  pc:   14	addr:  116	row:   0
# 	L2 HIT   pc:   14	addr:  116	row:   1
# 	L1 MISS  pc:   10	addr:  100	row:   0
# 	L2 HIT   pc:   10	addr:  100	row:   1
# 	L1 MISS  pc:   11	addr:  104	row:   0
# 	L2 HIT   pc:   11	addr:  104	row:   2
# 	L1 MISS  pc:   12	addr:  108	row:   0
# 	L2 HIT   pc:   12	addr:  108	row:   3
# 	L1 MISS  pc:   13	addr:  112	row:   0
# 	L2 HIT   pc:   13	addr:  112	row:   0
# 	L1 MISS  pc:   14	addr:  116	row:   0
# 	L2 HIT   pc:   14	addr:  116	row:   1
# 	L1 MISS  pc:   10	addr:  100	row:   0
# 	L2 HIT   pc:   10	addr:  100	row:   1
# 	L1 MISS  pc:   11	addr:  104	row:   0
# 	L2 HIT   pc:   11	addr:  104	row:   2
# 	L1 MISS  pc:   12	addr:  108	row:   0
# 	L2 HIT   pc:   12	addr:  108	row:   3
# 	L1 MISS  pc:   13	addr:  112	row:   0
# 	L2 HIT   pc:   13	addr:  112	row:   0
# 	L1 MISS  pc:   14	addr:  116	row:   0
# 	L2 HIT   pc:   14	addr:  116	row:   1
# 	L1 MISS  pc:   10	addr:  100	row:   0
# 	L2 HIT   pc:   10	addr:  100	row:   1
# 	L1 MISS  pc:   11	addr:  104	row:   0
# 	L2 HIT   pc:   11	addr:  104	row:   2
# 	L1 MISS  pc:   12	addr:  108	row:   0
# 	L2 HIT   pc:   12	addr:  108	row:   3
# 	L1 MISS  pc:   13	addr:  112	row:   0
# 	L2 HIT   pc:   13	addr:  112	row:   0
# 	L1 MISS  pc:   14	addr:  116	row:   0
# 	L2 HIT   pc:   14	addr:  116	row:   1
# 	L1 MISS  pc:   10	addr:  100	row:   0
# 	L2 HIT   pc:   10	addr:  100	row:   1
# 	L1 MISS  pc:   11	addr:  104	row:   0
# 	L2 HIT   pc:   11	addr:  104	row:   2
# 	L1 MISS  pc:   12	addr:  108	row:   0
# 	L2 HIT   pc:   12	addr:  108	row:   3
# 	L1 MISS  pc:   13	addr:  112	row:   0
# 	L2 HIT   pc:   13	addr:  112	row:   0
# 	L1 MISS  pc:   14	addr:  116	row:   0
# 	L2 HIT   pc:   14	addr:  116	row:   1
# 	L1 MISS  pc:   10	addr:  100	row:   0
# 	L2 HIT   pc:   10	addr:  100	row:   1
# 	L1 MISS  pc:   11	addr:  104	row:   0
# 	L2 HIT   pc:   11	addr:  104	row:   2
# 	L1 MISS  pc:   12	addr:  108	row:   0
# 	L2 HIT   pc:   12	addr:  108	row:   3
# 	L1 MISS  pc:   13	addr:  112	row:   0
# 	L2 HIT   pc:   13	addr:  112	row:   0
# 	L1 MISS  pc:   14	addr:  116	row:   0
# 	L2 HIT   pc:   14	addr:  116	row:   1
# 	L1 MISS  pc:   10	addr:  100	row:   0
# 	L2 HIT   pc:   10	addr:  100	row:   1
# 	L1 MISS  pc:   11	addr:  104	row:   0
# 	L2 HIT   pc:   11	addr:  104	row:   2
# 	L1 MISS  pc:   12	addr:  108	row:   0
# 	L2 HIT   pc:   12	addr:  108	row:   3
# 	L1 MISS  pc:   13	addr:  112	row:   0
# 	L2 HIT   pc:   13	addr:  112	row:   0
# 	L1 MISS  pc:   14	addr:  116	row:   0
# 	L2 HIT   pc:   14	addr:  116	row:   1
# 	L1 MISS  pc:   10	addr:  100	row:   0
# 	L2 HIT   pc:   10	addr:  100	row:   1
# 	L1 MISS  pc:   11	addr:  104	row:   0
# 	L2 HIT   pc:   11	addr:  104	row:   2
# 	L1 MISS  pc:   12	addr:  108	row:   0
# 	L2 HIT   pc:   12	addr:  108	row:   3
# 	L1 MISS  pc:   13	addr:  112	row:   0
# 	L2 HIT   pc:   13	addr:  112	row:   0
# 	L1 MISS  pc:   14	addr:  116	row:   0
# 	L2 HIT   pc:   14	addr:  116	row:   1
# 
