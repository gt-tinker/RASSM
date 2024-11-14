
typename="csr-32"

# check if environment variable RASSM_HOME is set
if [ -z "$RASSM_HOME" ]; then
    echo "RASSM_HOME is not set. Please set it to the path of the RASSM repository"
    exit 1
fi

# make the output directory at RASSM_HOME/output
output_dir="$RASSM_HOME/logs"
mkdir -p $output_dir

# check if environment variable RASSM_DATASET is set
if [ -z "$RASSM_DATASET" ]; then
    echo "RASSM_DATASET is not set. Please set it to the path of the dataset"
    exit 1
fi

# check if RASMM_BUILD is set
if [ -z "$RASSM_BUILD" ]; then
    echo "RASSM_BUILD is not set. Please set it to the path of the build directory"
    exit 1
fi

mpath=$RASSM_DATASET
epath=$RASSM_BUILD

feature=$1

executable="${epath}/rassm"

# find number of physical cores on the machine
cores=$(lscpu | grep "Core(s) per socket" | awk '{print $4}')
# find number of sockets on the machine
sockets=$(lscpu | grep "Socket(s)" | awk '{print $2}')
# set cores to the number of physical cores across all sockets
cores=$(($cores * $sockets))
# set threads to the number of physical cores across all sockets
threads=$cores

# find the L2 cache size of the machine
l2_cache=$(lscpu | grep "L2 cache" | awk '{print $3}')
# convert the L2 cache size to KB
l2_cache=$(echo $l2_cache | sed 's/[^0-9]*//g')
# find the number of ways in the L2 cache using l2_cache if each way is 64KB
l2_ways=$(($l2_cache / 64))

# print the configuration
echo "Run configuration"
echo "Threads: $threads"
echo "Cores: $cores"
echo "L2 Cache: $l2_cache KB"
echo "L2 Ways: $l2_ways"
echo "Matrix path: $mpath"

greedy=0
residue=0
nruns=50
kernel="spmm"
layers=4


let cachesize=$l2_cache*1024

opath="${output_dir}/K${feature}/${typename}"

logfile="${opath}/thread_${threads}.log"
errfile="${opath}/thread_${threads}.err"
touch ${logfile}
touch ${errfile}

tfile="${opath}/run.out"
efile="${opath}/run.err"
for matrix in $mpath/*
do
    matrix_name=$(basename ${matrix})
    ofilepath="${opath}/thread_${threads}/${matrix_name}"

    if [[ -d ${ofilepath} ]]; then
        continue
    fi

    mkdir -p ${ofilepath}
    echo ${matrix_name}
    ofile="${ofilepath}/$typename.out"
    out_e_file="${ofilepath}/$typename.err"

    if test -f "${ofile}"; then
        rm ${ofile}
    fi
    matrix_path="${mpath}/${matrix_name}/${matrix_name}.mtx"
    OMP_NUM_THREADS=${threads} OMP_PROC_BIND=true OMP_PLACES="cores(${cores})" ${executable} --m ${matrix_path} --feature ${feature} --threads ${threads} --numruns ${nruns} --layers ${layers} --residue ${residue} --greedy ${greedy} --kernel ${kernel} --type "CSR_32" 2>$efile >$tfile

    grep "Median Time" $tfile >> ${ofile}
    grep "GFLOPS" $tfile >> ${ofile}
    echo $'\n\n' >> ${ofile}

    cat $tfile >> $logfile
    echo $matrix >> ${errfile}
    cat $efile >> ${errfile}
    cat $efile >> ${out_e_file}
done

rm $tfile
rm $efile

