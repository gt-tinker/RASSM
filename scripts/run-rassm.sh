
mpath="/nethome/ajain324/USERSCRATCH/ssData/large_dataset_40M"
# epath="/nethome/ajain324/Resgemm-timing/amd/build.stats/spmm-asplos-final/atm-special"
# epath="/nethome/ajain324/Resgemm-timing/amd/build.stats/spmm-asplos-final/rassm-atm"
epath="/nethome/ajain324/Resgemm-timing/amd/build.stats/asplos-camera-ready/atm"

ename=$1
feature=$2
split=$3
sort_tiles=$4
squeeze_tiles=$5
temporal_output=$6
temporal_input=$7
oi_aware=$8
machine=$9
serialize=${10}
Ri=${11:-64}
Rj=${12:-64}

executable="${epath}/${ename}"

greedy=1
cores=64
threads=64
nruns=50
kernel="spmm"

layers=4

# for csstr in 256 384; do
for csstr in 512; do
    let cachesize=csstr*1024

    opath="/nethome/ajain324/USERSCRATCH/spmm-asplos-camera-ready/$machine/K${feature}/atm-perf-greedy-${greedy}-Ri-$Ri-Rj-$Rj-split-${split}-cache-${csstr}-sort-${sort_tiles}-squeeze-${squeeze_tiles}-tempout-${temporal_output}-tempin-${temporal_input}-oiaware-${oi_aware}-special-perf-${ename}"

    # opath="/nethome/ajain324/USERSCRATCH/spmm-asplos-camera-ready/$machine/K${feature}/atm-sweep-greedy-${greedy}-Ri-$Ri-Rj-$Rj-split-${split}-cache-${csstr}-sort-${sort_tiles}-squeeze-${squeeze_tiles}-tempout-${temporal_output}-tempin-${temporal_input}-oiaware-${oi_aware}-special-perf-${ename}"
    
    rpath="/nethome/ajain324/USERSCRATCH/spmm-asplos-final/residues-special/$machine/K${feature}/atm-greedy-${greedy}-split-${split}-cache-${csstr}-sort-${sort_tiles}-squeeze-${squeeze_tiles}-tempout-${temporal_output}-tempin-${temporal_input}-oiaware-${oi_aware}"

    mkdir -p $rpath

    tfile="${opath}/run.out"
    efile="${opath}/run.err"

    logfile="${opath}/thread_${threads}_Ri_${Ri}.log"
    errfile="${opath}/thread_${threads}_Ri_${Ri}.err"

    # Ti="adaptive"
    echo "" > ${logfile}

    for matrix in $mpath/*
    do
        matrix_name=$(basename ${matrix})
        ofilepath="${opath}/thread_${threads}/${matrix_name}"

        if [[ -d ${ofilepath} ]]; then
            continue
        fi

        rfile="${rpath}/${matrix_name}.panels"
        echo $rfile

        mkdir -p ${ofilepath}
        echo ${matrix_name}
        ofile="${ofilepath}/${Ti}_${Ri}.out"
        out_e_file="${ofilepath}/${Ti}_${Ri}.err"

        if test -f "${ofile}"; then
            rm ${ofile}
        fi
        matrix_path="${mpath}/${matrix_name}/${matrix_name}.mtx"
        OMP_NUM_THREADS=${threads} OMP_PROC_BIND=true OMP_PLACES="cores(${cores})" ${executable} --m ${matrix_path} --feature ${feature} --threads ${threads} --numruns ${nruns} --Ri ${Ri} --Rj ${Rj} --targCache ${cachesize} --layers ${layers} --greedy ${greedy} --cache-split ${split} --sort-tiles ${sort_tiles} --temporal-output ${temporal_output} --temporal-input ${temporal_input} --oi-aware ${oi_aware}  --serialize-panels ${serialize} --panel-file ${rfile} --kernel ${kernel}  2>$efile >$tfile

        sed -n '/START READING/,/STOP READING/p' $tfile > "${ofilepath}/panels.log"
        sed -n '/START CDF/,/STOP CDF/p' $tfile > "${ofilepath}/cdf.log"
        sed -n '/START PANEL PRINT/,/STOP PANEL PRINT/p' $tfile > $"${ofilepath}/panel_stats.log"
        sed -n '/###START PER PANEL TIME PARSING###/,/###STOP PER PANEL TIME PARSING###/p' $tfile > "${ofilepath}/panel_times.log"
        sed -n '/START_TILE_HISTOGRAM/,/STOP_TILE_HISTOGRAM/p' $tfile > "${ofilepath}/tile_histogram.log"
        sed -n '/START_TILE_NNZ_HISTOGRAM/,/STOP_TILE_NNZ_HISTOGRAM/p' $tfile > "${ofilepath}/tile_nnz_histogram.log"

        grep "Median Time" $tfile >> ${ofile}
        grep "GFLOPS" $tfile >> ${ofile}
        grep "PAPI_L1_DCA" $tfile >> ${ofile}
        grep "PAPI_L1_DCM" $tfile >> ${ofile}
        grep "PAPI_L2_DCM" $tfile >> ${ofile}
        grep "UNC_L3_MISSES" $tfile >> ${ofile}
        grep "UNC_L3_REQUESTS" $tfile >> ${ofile}
        grep "MEM_IO_RMT" $tfile >> ${ofile}
        grep "EXT_CACHE_RMT" $tfile >> ${ofile}
        grep "MEM_IO_LCL" $tfile >> ${ofile}
        grep "EXT_CACHE_LCL" $tfile >> ${ofile}
        grep "Global active cols count" $tfile >> ${ofile}
        grep "Mean Time" $tfile >> ${ofile}
        grep "Std Dev" $tfile >> ${ofile}
        grep "RAW" $tfile >> ${ofile}
        grep "GEN TIME" $tfile >> ${ofile}
        echo $'\n\n' >> ${ofile}

        # Grep from errfile
        grep "Residue Generation Time" $efile >> $ofile
        grep "Total range generation time:" $efile >> $ofile
        grep "Total panels generated" $efile >> $ofile
        grep "Total tiles created" $efile >> $ofile
        grep "Count of Band Panels detected" $efile >> $ofile
        grep "Greedy Panel Generation Time" $efile >> $ofile
        grep "Total tile generation time" $efile >> $ofile
        grep "Special Row Count" $efile >> $ofile
        grep "Total FLOP Count" $efile >> $ofile


        cat $tfile >> $logfile
        echo $matrix >> ${errfile}
        cat $efile >> ${errfile}
        cat $efile >> ${out_e_file}

    done
done

rm $tfile
rm $efile

