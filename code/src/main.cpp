
#include "combination.h"
#include "config.h"
#include "experiments.h"
#include "global.h"
#include "Reader.h"
#include "Residue.h"
// #include "Spdmm.h"
#include "utils/Statistics.h"
#include "utils/util.h"

#include <boost/program_options.hpp>

#include <chrono>
#include <iostream>
#include <omp.h>
#include <vector>


int main(int argc, char *argv[])
{

    #ifdef INTEL_COMPILER
        std::cout << "Using an Intel compiler" << std::endl;
    #endif

    boost::program_options::options_description desc{"Options"};
    desc.add_options()
        ("mtx,m", boost::program_options::value<std::string>()->default_value(""), "matrix market file path/name")
        ("data", boost::program_options::value<std::string>()->default_value("../data/residues/"), "data store base path")
        ("kernel", boost::program_options::value<std::string>()->default_value("spmm"), "kernel to run")

        ("Ti", boost::program_options::value<ITYPE>()->default_value(512), "I dimension tile size")
        ("Tj", boost::program_options::value<ITYPE>()->default_value(512), "J dimension tile size")
        ("Tk", boost::program_options::value<ITYPE>()->default_value(512), "K dimension tile size")
        ("partitions", boost::program_options::value<ITYPE>()->default_value(2), "Number of partition for the SPLIT CSR sparse format")
        ("feature,f", boost::program_options::value<ITYPE>()->default_value(128), "Default number of columns in the dense matrix")
        ("numruns, n", boost::program_options::value<ITYPE>()->default_value(10), "Number of runs to perform for each multiplication")
        ("perf,p", boost::program_options::value<bool>()->default_value(false), "Perform the perf check")
        ("testsize", boost::program_options::value<ITYPE>()->default_value(1024 * 1024 * 1024 / sizeof(double)), "DRAM test size")

        ("residue,r", boost::program_options::value<bool>()->default_value(true), "Run Residue Tile Creation")
        ("Ri", boost::program_options::value<ITYPE>()->default_value(64), "Residue matrix tile size is RxR")
        ("Rj", boost::program_options::value<ITYPE>()->default_value(64), "Residue matrix tile size is RxR")
        ("greedy", boost::program_options::value<bool>()->default_value(false), "Use the greedy algorithm for the residue matrix")
        ("squeeze-tiles", boost::program_options::value<bool>()->default_value(false), "Squeeze the tiles in each matrix panel")
        ("sort-tiles", boost::program_options::value<bool>()->default_value(true), "Sort the tiles in each matrix panel")
        ("panel-file", boost::program_options::value<std::string>()->default_value(""), "panel file path/name")
        ("serialize-panels", boost::program_options::value<bool>()->default_value(false), "serialize the panels to a file")
        ("temporal-input", boost::program_options::value<bool>()->default_value(false), "use temporal volume instead of spatial")
        ("temporal-output", boost::program_options::value<bool>()->default_value(false), "use temporal volume instead of spatial")
        ("oi-aware", boost::program_options::value<bool>()->default_value(false), "use temporal volume instead of spatial")

        ("stm", boost::program_options::value<std::string>()->default_value(""), "stm file path/name")

        ("nrows", boost::program_options::value<ITYPE>()->default_value(1024), "NROWS in band matrix")
        ("ncols", boost::program_options::value<ITYPE>()->default_value(1024), "NCOLS in band matrix")
        ("fixed-nnzs", boost::program_options::value<ITYPE>()->default_value(0), "Fixed NNZS, variable sized tile size")
        ("b", boost::program_options::value<ITYPE>()->default_value(-1), "half band size in matrix")
        ("sparsity", boost::program_options::value<double>()->default_value(1), "sparsity of the band matrix expressed as a fraction between (0, 1]")
        ("stride", boost::program_options::value<ITYPE>()->default_value(8), "Band Matrix experiment Stride (resolution)")
        ("staticTi", boost::program_options::value<bool>()->default_value(false), "enable static (disable adaptive) Ti builder")
        ("staticTk", boost::program_options::value<bool>()->default_value(false), "enable static (disable adaptive) Tk builder")
        ("schedule", boost::program_options::value<bool>()->default_value(false), "static vs load balanced panel scheduling")
        ("inplace", boost::program_options::value<bool>()->default_value(false), "build matrices without allocating extra array")
        ("csc", boost::program_options::value<bool>()->default_value(true), "enable csc panels in hybrid construction")
        ("csr", boost::program_options::value<bool>()->default_value(true), "enable csr panels in hybrid construction")
        ("special", boost::program_options::value<bool>()->default_value(true), "create csc only when csr does'nt fit")

        ("alpha", boost::program_options::value<double>()->default_value(5.0), "score impact of MAC operations")
        ("beta", boost::program_options::value<double>()->default_value(14.0), "score impact of data movement due to NAC count")

        ("looporder,l", boost::program_options::value<ITYPE>()->default_value(0), "Loop ordering for the multiplication [0 - kij; 1 - ikj; 2 - kji; 3 - jki]")
        ("threads,t", boost::program_options::value<ITYPE>()->default_value(1), "Number of parallel threads")
        ("layers", boost::program_options::value<ITYPE>()->default_value(10), "Number of layers of the dense matrices")
        ("debug,d", boost::program_options::value<bool>()->default_value(false), "Turn on debug print statements")
        ("targThreads", boost::program_options::value<ITYPE>()->default_value(20), "Number of threads in the target machine")
        ("targCache", boost::program_options::value<ITYPE>()->default_value(DEFAULT_LLC), "Number of bytes in the target machine cache")
        ("numways", boost::program_options::value<ITYPE>()->default_value(8), "Number of ways in the target machine cache")
        ("cache-split", boost::program_options::value<ITYPE>()->default_value(4), "Fraction of cache for the output dense array in the 2D tiling case")
        ("targLLC", boost::program_options::value<ITYPE>()->default_value(DEFAULT_LLC), "Num bytes in the target LLC per core")
        ("targFeature", boost::program_options::value<ITYPE>()->default_value(128), "width (ncols) of the input dense matrix")
        ("resolution", boost::program_options::value<ITYPE>()->default_value(1), "resolution of the residue matrix")
        ("chunkSize", boost::program_options::value<ITYPE>()->default_value(1), "omp chunk size")
        ("runcheck", boost::program_options::value<bool>()->default_value(false), "run correctness check");


    boost::program_options::variables_map options;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), options);
    boost::program_options::notify(options);

    bool residue = true;    // control residue matrix construction is enabled
    bool serialize_panels = false;

    bool runcheck = false;
    ITYPE Ti = 128, Tk, Tj, Ri = 64, Rj = 64, f, n, resolution;
    ITYPE num_partitions = 2;
    ITYPE nrows, ncols, b, stride, fixed_nnzs;
    double sparsity = 0.1;

    ITYPE target_cache_size;
    ITYPE cache_split;
    ITYPE target_LLC_size;
    ITYPE target_num_threads;
    ITYPE target_feature_size;
    ITYPE test_arr_size;
    const ITYPE vec_size = 32;

    ITYPE num_threads = 64; // keep num_threads as threads
    ITYPE layers = 1;
    ITYPE chunk_size = -1;
    bool perf_check = false;
    bool staticTi_builder = false;
    bool staticTk_builder = false;
    bool static_scheduler = false;
    bool build_inplace = false;

    // To control CSR and CSC panels in the adaptive dynamic construction methods
    bool enable_csc = true;
    bool enable_csr = true;
    bool csc_on_overflow_only = true;

    // Adaptive 2D control knobs
    bool greedy = false;
    bool sort_tiles = false;
    bool squeeze_tiles = false;
    bool temporal_input = false;
    bool temporal_output = false;
    bool oi_aware = false;

    std::string mtx_file;
    std::string data_path;
    std::string mtx_name;
    std::string stm_file;
    std::string kernel;
    std::string panel_file; // if reading in precomputed panels from command line

    for (const auto& option : options) {
        std::cout << option.first << ": ";
        auto &value = option.second.value();
        if (auto v = boost::any_cast<ITYPE>(&value)) {
            // Read in the tile size parameters, etc.
            if (option.first == "Ti") {
                Ti = *v;
            } else if (option.first == "Tj") {
                Tj = *v;
            } else if (option.first == "Tk") {
                Tk = *v;
            } else if (option.first == "partitions") {
                num_partitions = *v;
            } else if (option.first == "residue") {
                Ri = *v;
            } else if (option.first == "feature") {
                f = *v;
            } else if (option.first == "numruns") {
                n = *v;
            } else if (option.first == "targThreads") {
                target_num_threads = *v;
            } else if (option.first == "targCache") {
                target_cache_size = *v;
            } else if (option.first == "cache-split") {
                cache_split = *v;
            } else if (option.first == "targFeature") {
                target_feature_size = *v;
            } else if (option.first == "targLLC") {
                target_LLC_size = *v;
            } else if (option.first == "resolution") {
                resolution = *v;
            } else if (option.first == "chunkSize") {
                chunk_size = *v;
            } else if (option.first == "Ri") {
                Ri = *v;
            } else if (option.first == "Rj") {
                Rj = *v;
            } else if (option.first == "b") {
                b = *v;
            } else if (option.first == "nrows") {
                nrows = *v;
            } else if (option.first == "ncols") {
                ncols = *v;
            } else if (option.first == "fixed-nnzs") {
                fixed_nnzs = *v;
            } else if (option.first == "stride") {
                stride = *v;
            } else if (option.first == "layers") {
                layers = *v;
            } else if (option.first == "testsize") {
                test_arr_size = *v;
            } else if (option.first == "threads") {
                num_threads = *v;
            } else if (option.first == "numways") {
                CACHE_NUM_WAYS = *v;
            }
            std::cout << *v;
        } else if (auto v = boost::any_cast<bool>(&value)) {
            if (option.first == "residue") {
                residue = *v;
            } else if (option.first == "perf") {
                perf_check = *v;
            } else if (option.first == "runcheck") {
                runcheck = *v;
            } else if (option.first == "staticTi") {
                staticTi_builder = *v;
            } else if (option.first == "staticTk") {
                staticTk_builder = *v;
            } else if (option.first == "schedule") {
                static_scheduler = *v;
            } else if (option.first == "inplace") {
                build_inplace = *v;
            } else if (option.first == "debug") {
                global_debug = *v;
            } else if (option.first == "csr") {
                enable_csr = *v;
            } else if (option.first == "csc") {
                enable_csc = *v;
            } else if (option.first == "special") {
                csc_on_overflow_only = *v;
            } else if (option.first == "greedy") {
                greedy = *v;
            } else if (option.first == "sort-tiles") {
                sort_tiles = *v;
            } else if (option.first == "squeeze-tiles") {
                squeeze_tiles = *v;
            } else if (option.first == "temporal-input") {
                temporal_input = *v;
            } else if (option.first == "serialize-panels") {
                serialize_panels = *v;
            } else if (option.first == "temporal-output") {
                temporal_output = *v;
            } else if (option.first == "oi-aware") {
                oi_aware = *v;
            }
            std::cout << *v;
        } else if (auto v = boost::any_cast<std::string>(&value)) {
            if (option.first == "mtx") {
                mtx_file = *v;
                auto size_mtx_name = mtx_file.find_last_of(".") - mtx_file.find_last_of("/") - 1;
                auto start_mtx_name = mtx_file.find_last_of("/") + 1;
                mtx_name = mtx_file.substr(start_mtx_name, size_mtx_name);
            } else if (option.first == "data") {
                data_path = *v;
            } else if (option.first == "stm") {
                stm_file = *v;
            } else if (option.first == "kernel") {
                kernel = *v;
            } else if (option.first == "panel-file") {
                panel_file = *v;
            }
            std::cout << *v;
        } else if (auto v = boost::any_cast<double>(&value)) {
            if (option.first == "sparsity") {
                sparsity = *v;
            } else if (option.first == "alpha") {
                alpha_val = *v;
            } else if (option.first == "beta") {
                beta_val = *v;
            }
            std::cout << *v;
        } else {
            std::cout << "error";
        }
        std::cout << std::endl;
    }

    data_path += data_path.back() == '/' ? (std::to_string(Ri) + "/") : ("/" + std::to_string(Ri) + "/");
    data_path += data_path.back() == '/' ? (mtx_name + "/") : ("/" + mtx_name + "/");


    std::cout << "Matrix: " << mtx_name << std::endl;

    // omp_set_num_threads((int) 1);
    omp_set_num_threads( (int) num_threads );


    #ifdef DATA_MOVEMENT_EXPERIMENT
        // Flush about 80MiB per core of data for the flush
        init_cache_flush( DEFAULT_CACHE_FLUSH_SIZE * num_threads );

        // #define TIME_CACHE_FLUSH
        #ifdef TIME_CACHE_FLUSH
            stats_t<double, ITYPE> cache_flush_time_stats;
            for (ITYPE i = 0; i < n; i++) {
                auto start = std::chrono::high_resolution_clock::now();
                cache_flush();
                auto end = std::chrono::high_resolution_clock::now();
                cache_flush_time_stats.insert( std::chrono::duration<double>(end - start).count() );
            }
            cache_flush_time_stats.process();
            cache_flush_time_stats.print();
        #endif

    #endif // DATA_MOVEMENT_EXPERIMENT

    ITYPE num_panels = 0;
    ITYPE *num_panels_per_thread = nullptr;
    struct workitem *work_list = nullptr;
    ITYPE *processed_worklist = nullptr;
    std::pair<ITYPE, ITYPE> *pairs_worklist = nullptr;
    std::vector<struct panel_t> adaptive_2d_tiles;

    if (residue) {
        auto start_time = std::chrono::high_resolution_clock::now();

        // check if the matrix is in mtx format or smtx format
        if (mtx_file.find(".mtx") != std::string::npos) {
            read_mtx_matrix_into_arrays<TYPE, ITYPE>( mtx_file.c_str(), &global_locs, &global_vals, &global_nrows, &global_ncols, &global_nnzs );
        } else if (mtx_file.find(".smtx") != std::string::npos) {
            read_smtx_matrix_into_arrays<TYPE, ITYPE>( mtx_file.c_str(), &global_locs, &global_vals, &global_nrows, &global_ncols, &global_nnzs );
        } else {
            std::cout << "Unknown matrix format: " << mtx_file << std::endl;
            std::exit(EXIT_FAILURE);
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = (end_time - start_time);
        print_status("Matrix File Read Time %f\n", diff.count() );

        CSR<TYPE, ITYPE> *spm;
        CSC<TYPE, ITYPE> *csc;
        Residue<TYPE, ITYPE> *res;
        if (!(greedy == 0 && fixed_nnzs > 0)) {
            start_time = std::chrono::high_resolution_clock::now();
            spm = new CSR<TYPE, ITYPE>( global_nrows, global_ncols, global_nnzs, global_locs, global_vals, build_inplace );
            end_time = std::chrono::high_resolution_clock::now();
            print_status("Matrix Generation Time %f\n", diff.count() );

            start_time = std::chrono::high_resolution_clock::now();
            csc = new CSC<TYPE, ITYPE>( global_nrows, global_ncols, global_nnzs, global_locs, global_vals );
            end_time = std::chrono::high_resolution_clock::now();
            diff = (end_time - start_time);
            print_status("CSC Generation Time %f\n", diff.count() );

            start_time = std::chrono::high_resolution_clock::now();
            res = new Residue<TYPE, ITYPE>(spm, csc, Ri, Rj, vec_size, data_path, mtx_name, resolution, (temporal_input || temporal_output) );
            end_time = std::chrono::high_resolution_clock::now();
            diff = end_time - start_time;

            print_status( "Residue generated with rows: %d, cols: %d, nnzs: %d\n", res->nrows, res->ncols, res->nnz );

            print_status( "Residue Generation Time %f\n", diff.count() );
        }


        // input processing after reading matrix dimensions and command line arguments
        ITYPE max_panel_size = CEIL(spm->nrows, num_threads);
        Ti = MIN( Ti, max_panel_size );
        Tk = MIN(f, Tk);
        if (Tj == -1) {
            Tj = spm->ncols;
        }
        print_status("Running Residue Generation -- Ti: %d, Tk: %d, Tj: %d\n", Ti, Tk, Tj);

        #ifndef RUN_ADAPTIVE_2D_TILING



        #else // RUN_ADAPTIVE_2D_TILING

            auto start_time_2d_tiling = std::chrono::high_resolution_clock::now();

            if (!serialize_panels && panel_file != "") {
                print_status("Reading in precomputed panels from file: %s\n", panel_file.c_str());
                adaptive_2d_tiles.clear();
                std::ifstream panel_stream(panel_file, std::ios::binary);
                if (panel_stream.is_open()) {
                    deserialize_panel_vector(panel_stream, adaptive_2d_tiles);
                }
                panel_stream.close();
            } else {
                if (greedy) {
                    // adaptive_2d_tiles = res->adaptive_2d_tile_generator_greedy_Ti_greedy_Tj(f, target_cache_size, cache_split);
                    adaptive_2d_tiles = res->adaptive_2d_greedy_Ti_greedy_Tj_tile_generator(f, target_cache_size, cache_split, temporal_input, temporal_output, oi_aware);
                } else {
                    // adaptive_2d_tiles = res->adaptive_2d_tile_generator_fixed_Ti_greedy_Tj(Ti, f, target_cache_size);
                    print_status("Generating adaptive 2D tiles with fixed Ti: %d and Tj: %d\n", Ti, Tj);
                    // adaptive_2d_tiles = res->adaptive_2d_fixed_Ti_greedy_Tj_tile_generator(Ti, f, target_cache_size);

                    adaptive_2d_tiles.clear();
                }

                if (serialize_panels && panel_file != "") {
                    print_status("Serializing panels to file: %s\n", panel_file.c_str());
                    std::ofstream panel_stream(panel_file, std::ios::binary);
                    if (panel_stream.is_open()) {
                        serialize_panel_vector(panel_stream, adaptive_2d_tiles);
                    }
                    panel_stream.close();
                }
            }

            auto end_time_2d_tiling = std::chrono::high_resolution_clock::now();
            auto adaptive_2d_tile_generation_time = std::chrono::duration<double>(end_time_2d_tiling - start_time_2d_tiling).count();
            print_status("Total tile generation time: %f\n", adaptive_2d_tile_generation_time);

            // identify_dense_panels<TYPE, ITYPE>(adaptive_2d_tiles, target_cache_size);

            if (squeeze_tiles) {
                ITYPE num_tiles_before_squeezing = 0;
                for (ITYPE i = 0; i < adaptive_2d_tiles.size(); i++) {
                    num_tiles_before_squeezing += adaptive_2d_tiles[i].tiles.size();
                }
                print_status("Number of tiles before squeezing: %d\n", num_tiles_before_squeezing);
                ITYPE num_tiles_squeezed = combine_generated_tiles<TYPE, ITYPE>( adaptive_2d_tiles, target_cache_size );
                print_status("Number of tiles squeezed: %d\n", num_tiles_squeezed);

                ITYPE num_tiles_after_squeezing = 0;
                for (ITYPE i = 0; i < adaptive_2d_tiles.size(); i++) {
                    num_tiles_after_squeezing += adaptive_2d_tiles[i].tiles.size();
                    std::cout << "Panel: " << i << " -- Num Tiles: " << adaptive_2d_tiles[i].tiles.size() << std::endl;
                }
                print_status("Number of tiles after squeezing: %d\n", num_tiles_after_squeezing);
            }

            if (sort_tiles) {   // sort tiles based on active rows and nnzs to break tiles
                sort_generated_tiles<TYPE, ITYPE>( adaptive_2d_tiles );
            }


        #endif // RUN_ADAPTIVE_2D_TILING

        // release memory used by the residue matrix
        delete csc;
        delete spm;
        delete res;

    } else {  // not running residue matrix tile generation
        // ITYPE nnzs;
        // parse_matrix_header<TYPE, ITYPE>( mtx_file.c_str(), &nrows, &ncols, &nnzs );
        parse_matrix_header<TYPE, ITYPE>( mtx_file.c_str(), &global_nrows, &global_ncols, &global_nnzs );
        ITYPE max_panel_size = CEIL(global_nrows, num_threads);
        Ti = MIN( Ti, max_panel_size );
        Tk = MIN(f, Tk);
        if ( Tj == -1 ) {
            Tj = global_ncols;
        }
        print_status("Non-Residue Run with Ti: %d, Tj: %d, Tk: %d\n", Ti, Tj, Tk);
    }

    size_t total_flop_count = ((size_t) global_nnzs) * ((size_t) 2) * ((size_t) f);
    print_status("Total FLOP Count: %ld\n", total_flop_count);

    #ifdef RUN_CSF_CHARACTERIZATION
        // generate_adaptive_matrix_histogram<TYPE, ITYPE>( adaptive_2d_tiles );
        {
            ITYPE nrows, ncols, nnzs;
            std::pair<ITYPE, ITYPE> *locs = nullptr;
            TYPE* vals = nullptr;
            CSF<TYPE, ITYPE> *S_csf = nullptr;
            if (global_locs && global_vals) {
                locs = global_locs;
                vals = global_vals;
                nrows = global_nrows;
                ncols = global_ncols;
                nnzs = global_nnzs;
            } else {
                read_mtx_matrix_into_arrays(stm_file.c_str(), &locs, &vals, &nrows, &ncols, &nnzs);
                assert( locs && vals && (nnzs > 0) && (ncols > 0) && (nnzs > 0) && "Could not read mtx file" );
            }
            ITYPE *C1, *C2, *C3, *C4;
            ITYPE num_csf_panels = 0;
            if ( adaptive_2d_tiles.size() > 0 ) {
                print_status("[CSF] Building rassm csf matrix\n");
                num_csf_panels = generate_coo_representation_rassm(nnzs, locs, vals, adaptive_2d_tiles, &C1, &C2, &C3, &C4);
            } else {
                if (fixed_nnzs > 0) {
                    print_status("[CSF] Building fixed nnzs csf matrix\n");
                    num_csf_panels = generate_fixed_nnzs_coo_representation(nnzs, locs, vals, fixed_nnzs, &C1, &C2, &C3, &C4);
                } else {
                    print_status("[CSF] Building fixed size csf matrix\n");
                    num_csf_panels = generate_coo_representation(Ti, Tj, nnzs, locs, vals, &C1, &C2, &C3, &C4);
                }
            }
            S_csf = new CSF<TYPE, ITYPE>( nrows, ncols, nnzs, num_csf_panels, C1, C2, C3, C4, vals );
            print_tile_histogram( *S_csf, f );

        }

    #endif // RUN_CSF_CHARACTERIZATION

    #ifdef DATA_MOVEMENT_EXPERIMENT

        if (kernel == "spmm") {
            data_movement_experiment<TYPE, ITYPE>( mtx_file, f, Ti, Tj, Tk, n, num_threads, chunk_size, layers, num_panels, num_panels_per_thread, work_list, processed_worklist, num_partitions, &adaptive_2d_tiles, stm_file, fixed_nnzs );
        } else if (kernel == "sddmm") {
            data_movement_experiment_sddmm<TYPE, ITYPE>( mtx_file, f, Ti, Tj, Tk, n, num_threads, chunk_size, layers, num_panels, num_panels_per_thread, work_list, processed_worklist, num_partitions, &adaptive_2d_tiles, fixed_nnzs );
        } else if (kernel == "spmv") {
            // data_movement_experiment_spmv<TYPE, ITYPE>( mtx_file, f, Ti, Tj, Tk, n, num_threads, chunk_size, layers, num_panels, num_panels_per_thread, work_list, processed_worklist, num_partitions, &adaptive_2d_tiles );
            // data_movement_experiment_spmv<TYPE, ITYPE>( mtx_file, f, Ti, Tj, Tk, n, num_threads, chunk_size, layers, num_panels, num_panels_per_thread, work_list, processed_worklist, num_partitions );
        } else {
            // do nothing -- maybe we make this a switch case with nop as an option
        }

    #endif

    // release global and other memory
    if ( global_locs ) { delete[] global_locs; }
    if ( global_vals ) { delete[] global_vals; }

    if ( num_panels_per_thread ) { delete[] num_panels_per_thread; }
    if ( processed_worklist ) { delete[] processed_worklist; }
    if ( work_list ) { delete[] work_list; }


    // release memory used for flushing the cache
    #ifdef DATA_MOVEMENT_EXPERIMENT
        free_cache_flush();
    #endif

    return 0;
}
