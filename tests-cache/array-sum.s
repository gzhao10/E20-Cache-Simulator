# We're going to sum an array of numbers.
# Myarray is a label pointing to the beginning of the array. The array
# can have any number of elements, but the last element must be zero.

# We use $3 as our running sum, and $1 as an index into the array.
# Both are initially zero.

# In the loop, we load the value at address myarray + $1. We add that
# value to the sum, and increment the index. We check if the value
# is zero, and if so, we quit. Otherwise, we iterate.


    movi $1, 0  # initialize counter
    movi $3, 0  # initialize sum

loop:
    lw $2, myarray($1)      # fetch on array item
    add $3, $3, $2          # add it to the sum
    addi $1, $1, 1          # increment counter
    jeq $2, $0, done        # is array item 0? if so, quit
    j loop                  # otherwise loop

done:
    halt                    # end the program

myarray:                    # this label denotes beginning of array
    .fill 434
    .fill 2342
    .fill 31924
    .fill 3
    .fill 85
    .fill 4307
    .fill 3474
    .fill 4944
    .fill 377
    .fill 4631
    .fill 942
    .fill 2400
    .fill 19
    .fill 1822
    .fill 660
    .fill 1461
    .fill 2189
    .fill 3118
    .fill 2495
    .fill 2128
    .fill 1338
    .fill 2973
    .fill 891
    .fill 1704
    .fill 2894
    .fill 3887
    .fill 486
    .fill 3032
    .fill 4925
    .fill 861
    .fill 3928
    .fill 3044
    .fill 826
    .fill 3255
    .fill 4574
    .fill 3593
    .fill 1175
    .fill 3258
    .fill 944
    .fill 293
    .fill 1690
    .fill 2314
    .fill 1748
    .fill 4882
    .fill 110
    .fill 2100
    .fill 234
    .fill 3288
    .fill 2738
    .fill 1174
    .fill 1178
    .fill 1029
    .fill 2314
    .fill 1448
    .fill 0

#--
#--
#--MACHINE CODE
# ram[0] = 16'b0010000010000000;		// movi $1,0
# ram[1] = 16'b0010000110000000;		// movi $3,0
# ram[2] = 16'b1000010100001000;		// loop: lw $2,myarray($1)
# ram[3] = 16'b0000110100110000;		// add $3,$3,$2
# ram[4] = 16'b0010010010000001;		// addi $1,$1,1
# ram[5] = 16'b1100100000000001;		// jeq $2,$0,done
# ram[6] = 16'b0100000000000010;		// j loop
# ram[7] = 16'b0100000000000111;		// done: halt 
# ram[8] = 16'b0000000110110010;		// myarray: .fill 434
# ram[9] = 16'b0000100100100110;		// .fill 2342
# ram[10] = 16'b0111110010110100;		// .fill 31924
# ram[11] = 16'b0000000000000011;		// .fill 3
# ram[12] = 16'b0000000001010101;		// .fill 85
# ram[13] = 16'b0001000011010011;		// .fill 4307
# ram[14] = 16'b0000110110010010;		// .fill 3474
# ram[15] = 16'b0001001101010000;		// .fill 4944
# ram[16] = 16'b0000000101111001;		// .fill 377
# ram[17] = 16'b0001001000010111;		// .fill 4631
# ram[18] = 16'b0000001110101110;		// .fill 942
# ram[19] = 16'b0000100101100000;		// .fill 2400
# ram[20] = 16'b0000000000010011;		// .fill 19
# ram[21] = 16'b0000011100011110;		// .fill 1822
# ram[22] = 16'b0000001010010100;		// .fill 660
# ram[23] = 16'b0000010110110101;		// .fill 1461
# ram[24] = 16'b0000100010001101;		// .fill 2189
# ram[25] = 16'b0000110000101110;		// .fill 3118
# ram[26] = 16'b0000100110111111;		// .fill 2495
# ram[27] = 16'b0000100001010000;		// .fill 2128
# ram[28] = 16'b0000010100111010;		// .fill 1338
# ram[29] = 16'b0000101110011101;		// .fill 2973
# ram[30] = 16'b0000001101111011;		// .fill 891
# ram[31] = 16'b0000011010101000;		// .fill 1704
# ram[32] = 16'b0000101101001110;		// .fill 2894
# ram[33] = 16'b0000111100101111;		// .fill 3887
# ram[34] = 16'b0000000111100110;		// .fill 486
# ram[35] = 16'b0000101111011000;		// .fill 3032
# ram[36] = 16'b0001001100111101;		// .fill 4925
# ram[37] = 16'b0000001101011101;		// .fill 861
# ram[38] = 16'b0000111101011000;		// .fill 3928
# ram[39] = 16'b0000101111100100;		// .fill 3044
# ram[40] = 16'b0000001100111010;		// .fill 826
# ram[41] = 16'b0000110010110111;		// .fill 3255
# ram[42] = 16'b0001000111011110;		// .fill 4574
# ram[43] = 16'b0000111000001001;		// .fill 3593
# ram[44] = 16'b0000010010010111;		// .fill 1175
# ram[45] = 16'b0000110010111010;		// .fill 3258
# ram[46] = 16'b0000001110110000;		// .fill 944
# ram[47] = 16'b0000000100100101;		// .fill 293
# ram[48] = 16'b0000011010011010;		// .fill 1690
# ram[49] = 16'b0000100100001010;		// .fill 2314
# ram[50] = 16'b0000011011010100;		// .fill 1748
# ram[51] = 16'b0001001100010010;		// .fill 4882
# ram[52] = 16'b0000000001101110;		// .fill 110
# ram[53] = 16'b0000100000110100;		// .fill 2100
# ram[54] = 16'b0000000011101010;		// .fill 234
# ram[55] = 16'b0000110011011000;		// .fill 3288
# ram[56] = 16'b0000101010110010;		// .fill 2738
# ram[57] = 16'b0000010010010110;		// .fill 1174
# ram[58] = 16'b0000010010011010;		// .fill 1178
# ram[59] = 16'b0000010000000101;		// .fill 1029
# ram[60] = 16'b0000100100001010;		// .fill 2314
# ram[61] = 16'b0000010110101000;		// .fill 1448
# ram[62] = 16'b0000000000000000;		// .fill 0
#--
#--
#--EXECUTION OUTPUT
# array-sum.bin --cache 4,1,1
# 	Cache L1 has size 4, associativity 1, blocksize 1, rows 4
# 	L1 MISS  pc:    2	addr:    8	row:   0
# 	L1 MISS  pc:    2	addr:    9	row:   1
# 	L1 MISS  pc:    2	addr:   10	row:   2
# 	L1 MISS  pc:    2	addr:   11	row:   3
# 	L1 MISS  pc:    2	addr:   12	row:   0
# 	L1 MISS  pc:    2	addr:   13	row:   1
# 	L1 MISS  pc:    2	addr:   14	row:   2
# 	L1 MISS  pc:    2	addr:   15	row:   3
# 	L1 MISS  pc:    2	addr:   16	row:   0
# 	L1 MISS  pc:    2	addr:   17	row:   1
# 	L1 MISS  pc:    2	addr:   18	row:   2
# 	L1 MISS  pc:    2	addr:   19	row:   3
# 	L1 MISS  pc:    2	addr:   20	row:   0
# 	L1 MISS  pc:    2	addr:   21	row:   1
# 	L1 MISS  pc:    2	addr:   22	row:   2
# 	L1 MISS  pc:    2	addr:   23	row:   3
# 	L1 MISS  pc:    2	addr:   24	row:   0
# 	L1 MISS  pc:    2	addr:   25	row:   1
# 	L1 MISS  pc:    2	addr:   26	row:   2
# 	L1 MISS  pc:    2	addr:   27	row:   3
# 	L1 MISS  pc:    2	addr:   28	row:   0
# 	L1 MISS  pc:    2	addr:   29	row:   1
# 	L1 MISS  pc:    2	addr:   30	row:   2
# 	L1 MISS  pc:    2	addr:   31	row:   3
# 	L1 MISS  pc:    2	addr:   32	row:   0
# 	L1 MISS  pc:    2	addr:   33	row:   1
# 	L1 MISS  pc:    2	addr:   34	row:   2
# 	L1 MISS  pc:    2	addr:   35	row:   3
# 	L1 MISS  pc:    2	addr:   36	row:   0
# 	L1 MISS  pc:    2	addr:   37	row:   1
# 	L1 MISS  pc:    2	addr:   38	row:   2
# 	L1 MISS  pc:    2	addr:   39	row:   3
# 	L1 MISS  pc:    2	addr:   40	row:   0
# 	L1 MISS  pc:    2	addr:   41	row:   1
# 	L1 MISS  pc:    2	addr:   42	row:   2
# 	L1 MISS  pc:    2	addr:   43	row:   3
# 	L1 MISS  pc:    2	addr:   44	row:   0
# 	L1 MISS  pc:    2	addr:   45	row:   1
# 	L1 MISS  pc:    2	addr:   46	row:   2
# 	L1 MISS  pc:    2	addr:   47	row:   3
# 	L1 MISS  pc:    2	addr:   48	row:   0
# 	L1 MISS  pc:    2	addr:   49	row:   1
# 	L1 MISS  pc:    2	addr:   50	row:   2
# 	L1 MISS  pc:    2	addr:   51	row:   3
# 	L1 MISS  pc:    2	addr:   52	row:   0
# 	L1 MISS  pc:    2	addr:   53	row:   1
# 	L1 MISS  pc:    2	addr:   54	row:   2
# 	L1 MISS  pc:    2	addr:   55	row:   3
# 	L1 MISS  pc:    2	addr:   56	row:   0
# 	L1 MISS  pc:    2	addr:   57	row:   1
# 	L1 MISS  pc:    2	addr:   58	row:   2
# 	L1 MISS  pc:    2	addr:   59	row:   3
# 	L1 MISS  pc:    2	addr:   60	row:   0
# 	L1 MISS  pc:    2	addr:   61	row:   1
# 	L1 MISS  pc:    2	addr:   62	row:   2
# 
# array-sum.bin --cache 16,1,4
# 	Cache L1 has size 16, associativity 1, blocksize 4, rows 4
# 	L1 MISS  pc:    2	addr:    8	row:   2
# 	L1 HIT   pc:    2	addr:    9	row:   2
# 	L1 HIT   pc:    2	addr:   10	row:   2
# 	L1 HIT   pc:    2	addr:   11	row:   2
# 	L1 MISS  pc:    2	addr:   12	row:   3
# 	L1 HIT   pc:    2	addr:   13	row:   3
# 	L1 HIT   pc:    2	addr:   14	row:   3
# 	L1 HIT   pc:    2	addr:   15	row:   3
# 	L1 MISS  pc:    2	addr:   16	row:   0
# 	L1 HIT   pc:    2	addr:   17	row:   0
# 	L1 HIT   pc:    2	addr:   18	row:   0
# 	L1 HIT   pc:    2	addr:   19	row:   0
# 	L1 MISS  pc:    2	addr:   20	row:   1
# 	L1 HIT   pc:    2	addr:   21	row:   1
# 	L1 HIT   pc:    2	addr:   22	row:   1
# 	L1 HIT   pc:    2	addr:   23	row:   1
# 	L1 MISS  pc:    2	addr:   24	row:   2
# 	L1 HIT   pc:    2	addr:   25	row:   2
# 	L1 HIT   pc:    2	addr:   26	row:   2
# 	L1 HIT   pc:    2	addr:   27	row:   2
# 	L1 MISS  pc:    2	addr:   28	row:   3
# 	L1 HIT   pc:    2	addr:   29	row:   3
# 	L1 HIT   pc:    2	addr:   30	row:   3
# 	L1 HIT   pc:    2	addr:   31	row:   3
# 	L1 MISS  pc:    2	addr:   32	row:   0
# 	L1 HIT   pc:    2	addr:   33	row:   0
# 	L1 HIT   pc:    2	addr:   34	row:   0
# 	L1 HIT   pc:    2	addr:   35	row:   0
# 	L1 MISS  pc:    2	addr:   36	row:   1
# 	L1 HIT   pc:    2	addr:   37	row:   1
# 	L1 HIT   pc:    2	addr:   38	row:   1
# 	L1 HIT   pc:    2	addr:   39	row:   1
# 	L1 MISS  pc:    2	addr:   40	row:   2
# 	L1 HIT   pc:    2	addr:   41	row:   2
# 	L1 HIT   pc:    2	addr:   42	row:   2
# 	L1 HIT   pc:    2	addr:   43	row:   2
# 	L1 MISS  pc:    2	addr:   44	row:   3
# 	L1 HIT   pc:    2	addr:   45	row:   3
# 	L1 HIT   pc:    2	addr:   46	row:   3
# 	L1 HIT   pc:    2	addr:   47	row:   3
# 	L1 MISS  pc:    2	addr:   48	row:   0
# 	L1 HIT   pc:    2	addr:   49	row:   0
# 	L1 HIT   pc:    2	addr:   50	row:   0
# 	L1 HIT   pc:    2	addr:   51	row:   0
# 	L1 MISS  pc:    2	addr:   52	row:   1
# 	L1 HIT   pc:    2	addr:   53	row:   1
# 	L1 HIT   pc:    2	addr:   54	row:   1
# 	L1 HIT   pc:    2	addr:   55	row:   1
# 	L1 MISS  pc:    2	addr:   56	row:   2
# 	L1 HIT   pc:    2	addr:   57	row:   2
# 	L1 HIT   pc:    2	addr:   58	row:   2
# 	L1 HIT   pc:    2	addr:   59	row:   2
# 	L1 MISS  pc:    2	addr:   60	row:   3
# 	L1 HIT   pc:    2	addr:   61	row:   3
# 	L1 HIT   pc:    2	addr:   62	row:   3
# 
# array-sum.bin --cache 32,2,4
# 	Cache L1 has size 32, associativity 2, blocksize 4, rows 4
# 	L1 MISS  pc:    2	addr:    8	row:   2
# 	L1 HIT   pc:    2	addr:    9	row:   2
# 	L1 HIT   pc:    2	addr:   10	row:   2
# 	L1 HIT   pc:    2	addr:   11	row:   2
# 	L1 MISS  pc:    2	addr:   12	row:   3
# 	L1 HIT   pc:    2	addr:   13	row:   3
# 	L1 HIT   pc:    2	addr:   14	row:   3
# 	L1 HIT   pc:    2	addr:   15	row:   3
# 	L1 MISS  pc:    2	addr:   16	row:   0
# 	L1 HIT   pc:    2	addr:   17	row:   0
# 	L1 HIT   pc:    2	addr:   18	row:   0
# 	L1 HIT   pc:    2	addr:   19	row:   0
# 	L1 MISS  pc:    2	addr:   20	row:   1
# 	L1 HIT   pc:    2	addr:   21	row:   1
# 	L1 HIT   pc:    2	addr:   22	row:   1
# 	L1 HIT   pc:    2	addr:   23	row:   1
# 	L1 MISS  pc:    2	addr:   24	row:   2
# 	L1 HIT   pc:    2	addr:   25	row:   2
# 	L1 HIT   pc:    2	addr:   26	row:   2
# 	L1 HIT   pc:    2	addr:   27	row:   2
# 	L1 MISS  pc:    2	addr:   28	row:   3
# 	L1 HIT   pc:    2	addr:   29	row:   3
# 	L1 HIT   pc:    2	addr:   30	row:   3
# 	L1 HIT   pc:    2	addr:   31	row:   3
# 	L1 MISS  pc:    2	addr:   32	row:   0
# 	L1 HIT   pc:    2	addr:   33	row:   0
# 	L1 HIT   pc:    2	addr:   34	row:   0
# 	L1 HIT   pc:    2	addr:   35	row:   0
# 	L1 MISS  pc:    2	addr:   36	row:   1
# 	L1 HIT   pc:    2	addr:   37	row:   1
# 	L1 HIT   pc:    2	addr:   38	row:   1
# 	L1 HIT   pc:    2	addr:   39	row:   1
# 	L1 MISS  pc:    2	addr:   40	row:   2
# 	L1 HIT   pc:    2	addr:   41	row:   2
# 	L1 HIT   pc:    2	addr:   42	row:   2
# 	L1 HIT   pc:    2	addr:   43	row:   2
# 	L1 MISS  pc:    2	addr:   44	row:   3
# 	L1 HIT   pc:    2	addr:   45	row:   3
# 	L1 HIT   pc:    2	addr:   46	row:   3
# 	L1 HIT   pc:    2	addr:   47	row:   3
# 	L1 MISS  pc:    2	addr:   48	row:   0
# 	L1 HIT   pc:    2	addr:   49	row:   0
# 	L1 HIT   pc:    2	addr:   50	row:   0
# 	L1 HIT   pc:    2	addr:   51	row:   0
# 	L1 MISS  pc:    2	addr:   52	row:   1
# 	L1 HIT   pc:    2	addr:   53	row:   1
# 	L1 HIT   pc:    2	addr:   54	row:   1
# 	L1 HIT   pc:    2	addr:   55	row:   1
# 	L1 MISS  pc:    2	addr:   56	row:   2
# 	L1 HIT   pc:    2	addr:   57	row:   2
# 	L1 HIT   pc:    2	addr:   58	row:   2
# 	L1 HIT   pc:    2	addr:   59	row:   2
# 	L1 MISS  pc:    2	addr:   60	row:   3
# 	L1 HIT   pc:    2	addr:   61	row:   3
# 	L1 HIT   pc:    2	addr:   62	row:   3
# 
# array-sum.bin --cache 4,1,2,8,4,2
# 	Cache L1 has size 4, associativity 1, blocksize 2, rows 2
# 	Cache L2 has size 8, associativity 4, blocksize 2, rows 1
# 	L1 MISS  pc:    2	addr:    8	row:   0
# 	L2 MISS  pc:    2	addr:    8	row:   0
# 	L1 HIT   pc:    2	addr:    9	row:   0
# 	L1 MISS  pc:    2	addr:   10	row:   1
# 	L2 MISS  pc:    2	addr:   10	row:   0
# 	L1 HIT   pc:    2	addr:   11	row:   1
# 	L1 MISS  pc:    2	addr:   12	row:   0
# 	L2 MISS  pc:    2	addr:   12	row:   0
# 	L1 HIT   pc:    2	addr:   13	row:   0
# 	L1 MISS  pc:    2	addr:   14	row:   1
# 	L2 MISS  pc:    2	addr:   14	row:   0
# 	L1 HIT   pc:    2	addr:   15	row:   1
# 	L1 MISS  pc:    2	addr:   16	row:   0
# 	L2 MISS  pc:    2	addr:   16	row:   0
# 	L1 HIT   pc:    2	addr:   17	row:   0
# 	L1 MISS  pc:    2	addr:   18	row:   1
# 	L2 MISS  pc:    2	addr:   18	row:   0
# 	L1 HIT   pc:    2	addr:   19	row:   1
# 	L1 MISS  pc:    2	addr:   20	row:   0
# 	L2 MISS  pc:    2	addr:   20	row:   0
# 	L1 HIT   pc:    2	addr:   21	row:   0
# 	L1 MISS  pc:    2	addr:   22	row:   1
# 	L2 MISS  pc:    2	addr:   22	row:   0
# 	L1 HIT   pc:    2	addr:   23	row:   1
# 	L1 MISS  pc:    2	addr:   24	row:   0
# 	L2 MISS  pc:    2	addr:   24	row:   0
# 	L1 HIT   pc:    2	addr:   25	row:   0
# 	L1 MISS  pc:    2	addr:   26	row:   1
# 	L2 MISS  pc:    2	addr:   26	row:   0
# 	L1 HIT   pc:    2	addr:   27	row:   1
# 	L1 MISS  pc:    2	addr:   28	row:   0
# 	L2 MISS  pc:    2	addr:   28	row:   0
# 	L1 HIT   pc:    2	addr:   29	row:   0
# 	L1 MISS  pc:    2	addr:   30	row:   1
# 	L2 MISS  pc:    2	addr:   30	row:   0
# 	L1 HIT   pc:    2	addr:   31	row:   1
# 	L1 MISS  pc:    2	addr:   32	row:   0
# 	L2 MISS  pc:    2	addr:   32	row:   0
# 	L1 HIT   pc:    2	addr:   33	row:   0
# 	L1 MISS  pc:    2	addr:   34	row:   1
# 	L2 MISS  pc:    2	addr:   34	row:   0
# 	L1 HIT   pc:    2	addr:   35	row:   1
# 	L1 MISS  pc:    2	addr:   36	row:   0
# 	L2 MISS  pc:    2	addr:   36	row:   0
# 	L1 HIT   pc:    2	addr:   37	row:   0
# 	L1 MISS  pc:    2	addr:   38	row:   1
# 	L2 MISS  pc:    2	addr:   38	row:   0
# 	L1 HIT   pc:    2	addr:   39	row:   1
# 	L1 MISS  pc:    2	addr:   40	row:   0
# 	L2 MISS  pc:    2	addr:   40	row:   0
# 	L1 HIT   pc:    2	addr:   41	row:   0
# 	L1 MISS  pc:    2	addr:   42	row:   1
# 	L2 MISS  pc:    2	addr:   42	row:   0
# 	L1 HIT   pc:    2	addr:   43	row:   1
# 	L1 MISS  pc:    2	addr:   44	row:   0
# 	L2 MISS  pc:    2	addr:   44	row:   0
# 	L1 HIT   pc:    2	addr:   45	row:   0
# 	L1 MISS  pc:    2	addr:   46	row:   1
# 	L2 MISS  pc:    2	addr:   46	row:   0
# 	L1 HIT   pc:    2	addr:   47	row:   1
# 	L1 MISS  pc:    2	addr:   48	row:   0
# 	L2 MISS  pc:    2	addr:   48	row:   0
# 	L1 HIT   pc:    2	addr:   49	row:   0
# 	L1 MISS  pc:    2	addr:   50	row:   1
# 	L2 MISS  pc:    2	addr:   50	row:   0
# 	L1 HIT   pc:    2	addr:   51	row:   1
# 	L1 MISS  pc:    2	addr:   52	row:   0
# 	L2 MISS  pc:    2	addr:   52	row:   0
# 	L1 HIT   pc:    2	addr:   53	row:   0
# 	L1 MISS  pc:    2	addr:   54	row:   1
# 	L2 MISS  pc:    2	addr:   54	row:   0
# 	L1 HIT   pc:    2	addr:   55	row:   1
# 	L1 MISS  pc:    2	addr:   56	row:   0
# 	L2 MISS  pc:    2	addr:   56	row:   0
# 	L1 HIT   pc:    2	addr:   57	row:   0
# 	L1 MISS  pc:    2	addr:   58	row:   1
# 	L2 MISS  pc:    2	addr:   58	row:   0
# 	L1 HIT   pc:    2	addr:   59	row:   1
# 	L1 MISS  pc:    2	addr:   60	row:   0
# 	L2 MISS  pc:    2	addr:   60	row:   0
# 	L1 HIT   pc:    2	addr:   61	row:   0
# 	L1 MISS  pc:    2	addr:   62	row:   1
# 	L2 MISS  pc:    2	addr:   62	row:   0
# 
# array-sum.bin --cache 4,1,1,64,4,4
# 	Cache L1 has size 4, associativity 1, blocksize 1, rows 4
# 	Cache L2 has size 64, associativity 4, blocksize 4, rows 4
# 	L1 MISS  pc:    2	addr:    8	row:   0
# 	L2 MISS  pc:    2	addr:    8	row:   2
# 	L1 MISS  pc:    2	addr:    9	row:   1
# 	L2 HIT   pc:    2	addr:    9	row:   2
# 	L1 MISS  pc:    2	addr:   10	row:   2
# 	L2 HIT   pc:    2	addr:   10	row:   2
# 	L1 MISS  pc:    2	addr:   11	row:   3
# 	L2 HIT   pc:    2	addr:   11	row:   2
# 	L1 MISS  pc:    2	addr:   12	row:   0
# 	L2 MISS  pc:    2	addr:   12	row:   3
# 	L1 MISS  pc:    2	addr:   13	row:   1
# 	L2 HIT   pc:    2	addr:   13	row:   3
# 	L1 MISS  pc:    2	addr:   14	row:   2
# 	L2 HIT   pc:    2	addr:   14	row:   3
# 	L1 MISS  pc:    2	addr:   15	row:   3
# 	L2 HIT   pc:    2	addr:   15	row:   3
# 	L1 MISS  pc:    2	addr:   16	row:   0
# 	L2 MISS  pc:    2	addr:   16	row:   0
# 	L1 MISS  pc:    2	addr:   17	row:   1
# 	L2 HIT   pc:    2	addr:   17	row:   0
# 	L1 MISS  pc:    2	addr:   18	row:   2
# 	L2 HIT   pc:    2	addr:   18	row:   0
# 	L1 MISS  pc:    2	addr:   19	row:   3
# 	L2 HIT   pc:    2	addr:   19	row:   0
# 	L1 MISS  pc:    2	addr:   20	row:   0
# 	L2 MISS  pc:    2	addr:   20	row:   1
# 	L1 MISS  pc:    2	addr:   21	row:   1
# 	L2 HIT   pc:    2	addr:   21	row:   1
# 	L1 MISS  pc:    2	addr:   22	row:   2
# 	L2 HIT   pc:    2	addr:   22	row:   1
# 	L1 MISS  pc:    2	addr:   23	row:   3
# 	L2 HIT   pc:    2	addr:   23	row:   1
# 	L1 MISS  pc:    2	addr:   24	row:   0
# 	L2 MISS  pc:    2	addr:   24	row:   2
# 	L1 MISS  pc:    2	addr:   25	row:   1
# 	L2 HIT   pc:    2	addr:   25	row:   2
# 	L1 MISS  pc:    2	addr:   26	row:   2
# 	L2 HIT   pc:    2	addr:   26	row:   2
# 	L1 MISS  pc:    2	addr:   27	row:   3
# 	L2 HIT   pc:    2	addr:   27	row:   2
# 	L1 MISS  pc:    2	addr:   28	row:   0
# 	L2 MISS  pc:    2	addr:   28	row:   3
# 	L1 MISS  pc:    2	addr:   29	row:   1
# 	L2 HIT   pc:    2	addr:   29	row:   3
# 	L1 MISS  pc:    2	addr:   30	row:   2
# 	L2 HIT   pc:    2	addr:   30	row:   3
# 	L1 MISS  pc:    2	addr:   31	row:   3
# 	L2 HIT   pc:    2	addr:   31	row:   3
# 	L1 MISS  pc:    2	addr:   32	row:   0
# 	L2 MISS  pc:    2	addr:   32	row:   0
# 	L1 MISS  pc:    2	addr:   33	row:   1
# 	L2 HIT   pc:    2	addr:   33	row:   0
# 	L1 MISS  pc:    2	addr:   34	row:   2
# 	L2 HIT   pc:    2	addr:   34	row:   0
# 	L1 MISS  pc:    2	addr:   35	row:   3
# 	L2 HIT   pc:    2	addr:   35	row:   0
# 	L1 MISS  pc:    2	addr:   36	row:   0
# 	L2 MISS  pc:    2	addr:   36	row:   1
# 	L1 MISS  pc:    2	addr:   37	row:   1
# 	L2 HIT   pc:    2	addr:   37	row:   1
# 	L1 MISS  pc:    2	addr:   38	row:   2
# 	L2 HIT   pc:    2	addr:   38	row:   1
# 	L1 MISS  pc:    2	addr:   39	row:   3
# 	L2 HIT   pc:    2	addr:   39	row:   1
# 	L1 MISS  pc:    2	addr:   40	row:   0
# 	L2 MISS  pc:    2	addr:   40	row:   2
# 	L1 MISS  pc:    2	addr:   41	row:   1
# 	L2 HIT   pc:    2	addr:   41	row:   2
# 	L1 MISS  pc:    2	addr:   42	row:   2
# 	L2 HIT   pc:    2	addr:   42	row:   2
# 	L1 MISS  pc:    2	addr:   43	row:   3
# 	L2 HIT   pc:    2	addr:   43	row:   2
# 	L1 MISS  pc:    2	addr:   44	row:   0
# 	L2 MISS  pc:    2	addr:   44	row:   3
# 	L1 MISS  pc:    2	addr:   45	row:   1
# 	L2 HIT   pc:    2	addr:   45	row:   3
# 	L1 MISS  pc:    2	addr:   46	row:   2
# 	L2 HIT   pc:    2	addr:   46	row:   3
# 	L1 MISS  pc:    2	addr:   47	row:   3
# 	L2 HIT   pc:    2	addr:   47	row:   3
# 	L1 MISS  pc:    2	addr:   48	row:   0
# 	L2 MISS  pc:    2	addr:   48	row:   0
# 	L1 MISS  pc:    2	addr:   49	row:   1
# 	L2 HIT   pc:    2	addr:   49	row:   0
# 	L1 MISS  pc:    2	addr:   50	row:   2
# 	L2 HIT   pc:    2	addr:   50	row:   0
# 	L1 MISS  pc:    2	addr:   51	row:   3
# 	L2 HIT   pc:    2	addr:   51	row:   0
# 	L1 MISS  pc:    2	addr:   52	row:   0
# 	L2 MISS  pc:    2	addr:   52	row:   1
# 	L1 MISS  pc:    2	addr:   53	row:   1
# 	L2 HIT   pc:    2	addr:   53	row:   1
# 	L1 MISS  pc:    2	addr:   54	row:   2
# 	L2 HIT   pc:    2	addr:   54	row:   1
# 	L1 MISS  pc:    2	addr:   55	row:   3
# 	L2 HIT   pc:    2	addr:   55	row:   1
# 	L1 MISS  pc:    2	addr:   56	row:   0
# 	L2 MISS  pc:    2	addr:   56	row:   2
# 	L1 MISS  pc:    2	addr:   57	row:   1
# 	L2 HIT   pc:    2	addr:   57	row:   2
# 	L1 MISS  pc:    2	addr:   58	row:   2
# 	L2 HIT   pc:    2	addr:   58	row:   2
# 	L1 MISS  pc:    2	addr:   59	row:   3
# 	L2 HIT   pc:    2	addr:   59	row:   2
# 	L1 MISS  pc:    2	addr:   60	row:   0
# 	L2 MISS  pc:    2	addr:   60	row:   3
# 	L1 MISS  pc:    2	addr:   61	row:   1
# 	L2 HIT   pc:    2	addr:   61	row:   3
# 	L1 MISS  pc:    2	addr:   62	row:   2
# 	L2 HIT   pc:    2	addr:   62	row:   3
# 
