#!/bin/bash

# check if RASSM_HOME is set
if [ -z "$RASSM_HOME" ]; then
    echo "RASSM_HOME is not set. Please set it to the path of the RASSM repository"
    exit 1
fi

# check if the RASSM_HOME directory exists
if [ ! -d "$RASSM_HOME/logs" ]; then
    echo "RASSM_HOME/logs does not exist. Please run the evaluation before using this script"
    exit 1
fi
indir="$RASSM_HOME/logs"
outdir="$RASSM_HOME/output"
mkdir -p $outdir

testdir="$indir/K256/rassm-*/thread_*/"
testdir=$(echo $testdir)

# check if the expanded test directory name exists
if [ ! -d "$testdir" ]; then
    echo "RASSM output directory $testdir does not exist. Please run the evaluation before using this script"
    exit 1
fi

# extract the number of threads from the test directory name
threads=$(echo $testdir | cut -d'/' -f9 | cut -d'_' -f2 | sort -u)

# extract the K value from the test directory name
kvalue=$(echo $testdir | cut -d'/' -f7 | cut -d'K' -f2 | sort -u)

# create a list of files to process by reading the directory names for all the directories in RASSM_HOME/logs/K256/rassm-ti-0-to-0/thread_32
files=$(ls $testdir)

# print the extraction configuration
echo "Extraction configuration"
echo "Threads: $threads"
echo "Feature value: $kvalue"
echo "Output directory: $outdir"

kernel="spmm"

for s in K${kvalue}; do
    algo=$(ls $indir/$s)

    for alg in ${algo[@]}; do

        for t in $threads; do
            thread="thread_$t"

            echo -e "Processing: \t$alg -- threads: $thread"

            # check if both cache and l3.cache exists for the alg using and operator
            if [[ -d $indir/$s/$alg.cache ]] && [[ -d $indir/$s/$alg.l3.cache ]]; then
                echo "Found cache and l3.cache for $alg"
            fi

            outname="${outdir}/${alg}_${t}core_${s}.csv"

            echo "mtx,runtime,gflops" > $outname

            for d in ${files[@]}; do
                temp=$(ls $indir/$s/$alg/$thread/$d/*.out)
                runfile=${temp[0]}

                if [ "$(uname)" == "Darwin" ] && [[ $(stat -f '%z' "$runfile") -lt 10 ]]; then
                    echo -e "\t\t\tERROR, ${d} did not run for ${alg} ${t}core ${s}"
                    echo "${d},0,0" >> $outname
                    continue
                elif [ "$(uname)" == "Linux" ] && [[ $(stat -c%s "$runfile") -lt 10 ]]; then
                    echo -e "\t\t\tERROR, ${d} did not run for ${alg} ${t}core ${s}"
                    echo "${d},0,0" >> $outname
                    continue
                fi

                medtime=$(sed '2q;d' ${runfile} | cut -d':' -f2 | tr -d '[:blank:] ')
                gflop=$( grep "GFLOPS" ${runfile} | cut -d':' -f2 | tr -d '[:blank:]')

                echo "${d},${medtime},${gflop}" >> $outname
            done

        done

    done

done
