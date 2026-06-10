#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate timeagent

# ========== Single Tasks (30) ==========
single_list=("household1"
"household2"
"household3"
"household4"
"household5"
"household6"
"household7"
"household8"
"household9"
"household10"
"cooking1"
"cooking2"
"cooking3"
"cooking4"
"cooking5"
"cooking6"
"cooking7"
"cooking8"
"cooking9"
"cooking10"
"lab1"
"lab2"
"lab3"
"lab4"
"lab5"
"lab6"
"lab7"
"lab8"
"lab9"
"lab10")

for item in "${single_list[@]}"
do
    echo "===== Running single task: $item ====="
    python LLM_test.py --taskName $item --lm custom --total_time 40 --save_path ./trajectory/custom_single --save_name $item
done

# ========== Dual Tasks (30) ==========
dual_list=('household1,household2'
'household2,household3'
'household3,household4'
'household4,household5'
'household5,household6'
'household6,household7'
'household7,household8'
'household8,household9'
'household9,household10'
'household10,household1'
'cooking1,cooking2'
'cooking2,cooking3'
'cooking3,cooking4'
'cooking4,cooking5'
'cooking5,cooking6'
'cooking6,cooking7'
'cooking7,cooking8'
'cooking8,cooking9'
'cooking9,cooking10'
'cooking10,cooking1'
'lab1,lab2'
'lab2,lab3'
'lab3,lab4'
'lab4,lab5'
'lab5,lab6'
'lab6,lab7'
'lab7,lab8'
'lab8,lab9'
'lab9,lab10'
'lab10,lab1')

for item in "${dual_list[@]}"
do
    echo "===== Running dual task: $item ====="
    python LLM_test.py --taskName $item --lm custom --total_time 40 --save_path ./trajectory/custom_dual --save_name $item
done

# ========== Triple Tasks (30) ==========
triple_list=('household1,household2,household3'
'household2,household3,household4'
'household3,household4,household5'
'household4,household5,household6'
'household5,household6,household7'
'household6,household7,household8'
'household7,household8,household9'
'household8,household9,household10'
'household9,household10,household1'
'household10,household1,household2'
'cooking1,cooking2,cooking3'
'cooking2,cooking3,cooking4'
'cooking3,cooking4,cooking5'
'cooking4,cooking5,cooking6'
'cooking5,cooking6,cooking7'
'cooking6,cooking7,cooking8'
'cooking7,cooking8,cooking9'
'cooking8,cooking9,cooking10'
'cooking9,cooking10,cooking1'
'cooking10,cooking1,cooking2'
'lab1,lab2,lab3'
'lab2,lab3,lab4'
'lab3,lab4,lab5'
'lab4,lab5,lab6'
'lab5,lab6,lab7'
'lab6,lab7,lab8'
'lab7,lab8,lab9'
'lab8,lab9,lab10'
'lab9,lab10,lab1'
'lab10,lab1,lab2')

for item in "${triple_list[@]}"
do
    echo "===== Running triple task: $item ====="
    python LLM_test.py --taskName $item --lm custom --total_time 40 --save_path ./trajectory/custom_triple --save_name $item
done

echo "===== All 90 tasks completed ====="
