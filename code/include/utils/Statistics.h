#ifndef STATISTICS_H
#define STATISTICS_H

#include <cstring>
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include "util.h"

#define DEFAULT_CAPACITY 100

template <typename T, typename ITYPE>
class stats_t {

public:
    std::string name;
    ITYPE id = 0;

    ITYPE capacity = 0; // Max possible number of instances
    ITYPE size = 0;
    bool processed = false;

    T *records = nullptr; // Raw data
    T *sorted_records = nullptr;

    // Cacluated statistics
    T mean = 0;
    T median = 0;
    T std_dev = 0;

    // Based on the sorted array
    T min = 0;
    T max = 0;
    ITYPE zero_count = 0;
    ITYPE num_less_than_mean = 0;
    ITYPE num_greater_than_mean = 0;

    // Median indexA and indexB for co-relating events
    int median_index_A;
    int median_index_B;

    stats_t(std::string name, ITYPE capacity, ITYPE id = -1) : capacity(capacity), name(name), id(id)
    {
        if (capacity > 0) {
            this->records = new T[capacity]();
            this->sorted_records = new T[capacity]();
            this->size = 0;

            this->zero_count = 0;
            this->num_greater_than_mean = 0;
            this->num_less_than_mean = 0;
        }
    }

    stats_t(std::string name) : name(name)
    {
        this->capacity = DEFAULT_CAPACITY;
        this->records = new T[this->capacity]();
        this->sorted_records = new T[this->capacity]();
        this->size = 0;
        this->zero_count = 0;
        this->num_greater_than_mean = 0;
        this->num_less_than_mean = 0;
    }

    stats_t(ITYPE capacity = DEFAULT_CAPACITY)
    {
        this->capacity = capacity;
        this->records = new T[this->capacity]();
        this->sorted_records = new T[this->capacity]();
        this->size = 0;
        this->zero_count = 0;
        this->num_greater_than_mean = 0;
        this->num_less_than_mean = 0;
    }

    ~stats_t()
    {
        if (this->records) {
            delete[] this->records;
        }

        if (this->sorted_records) {
            delete[] this->sorted_records;
        }
    }

    void resize(ITYPE new_capacity)
    {
        if (this->capacity < new_capacity) {
            if (this->records) { delete[] this->records; }
            if (this->sorted_records) { delete[] this->sorted_records; }
            this->capacity = new_capacity;
            this->size = 0;
            this->records = new T[this->capacity];
            this->sorted_records = new T[this->capacity];
        }
    }

    void insert(T entry)
    {
        // assert (capacity > 0 && size < capacity && "stats object is full");
        if (size == capacity) {
            T *temp = new T[this->capacity * 2]();
            std::memcpy( temp, this->records, sizeof(T) * this->capacity );
            delete[] this->records;
            this->records = temp;
            this->capacity *= 2;

            delete[] this->sorted_records;
            this->sorted_records = new T[this->capacity]();
        }

        if (entry == 0) {
            this->zero_count++;
        }
        this->records[this->size++] = entry;
    }

    void process()
    {
        if (size <= 0) {
            return;
        }
        // assert (size > 0 && "process called on an empty stats object");
        std::memcpy(sorted_records, records, sizeof(T) * size);
        this->median = find_median<T>( sorted_records, size );
        this->mean = find_mean<T>( sorted_records, size );
        this->std_dev = find_std_dev( sorted_records, size, this->mean );
        this->max = sorted_records[ size - 1 ];
        this->min = sorted_records[0];

        this->num_less_than_mean = 0;
        this->zero_count = 0;
        for ( ITYPE i = 0; i < size; i++ ) {
            if ( sorted_records[i] < mean ) {
                this->num_less_than_mean++;
            }
            if ( sorted_records[i] == 0 ) {
                this->zero_count++;
            }
        }
        this->num_greater_than_mean = this->size - this->num_less_than_mean;

        T median_val_A = this->sorted_records[ (size-1) / 2 ];
        T median_val_B = this->sorted_records[ size / 2 ];

        for (ITYPE i = 0; i < size; i++) {
            if (records[i] == median_val_A) {
                median_index_A = i;
            }
            if (records[i] == median_val_B) {
                median_index_B = i;
            }
        }
    }

    // Method to print the array
    void print()
    {
        std::cout << this->name << " : {" // << (this->id >= 0 ? ( this->id << " -- " ) : "")
                    << "Mean: " << this->mean << " -- " << "Median: " << this->median << " -- "
                        << "std_dev: " << this->std_dev << " -- " << "Min: " << this->min << " -- "
                            << "Max: " << this->max << " -- " << "zero_cnt: " << this->zero_count << " -- "
                                << "num_less_mean: " << this->num_less_than_mean << " -- " << "num_grtr_mean: " << this->num_greater_than_mean << " -- ";
                                    print_arr<T>( this->records, this->size, "Raw", 1 );
                                    std::cout << "}";


        std::cout << std::endl;
    }

    T sum()
    {
        T sum = 0;
        for ( ITYPE i = 0; i < this->size; i++ ) {
            sum += this->records[i];
        }
        return sum;
    }

    // Method to print just the summary of the stats object
    void print_summary()
    {
        std::cout << this->name << " -- " << "Median: " << this->median << " -- " << "Mean: " << this->mean
                    << " -- " << "Min: " << this->min << " -- " << "Max: " << this->max << std::endl;
    }

    void reset()
    {
        this->size = 0;
        this->zero_count = 0;
        this->num_greater_than_mean = 0;
        this->num_less_than_mean = 0;
    }

};

template<typename T, typename ITYPE>
void stats_to_csv(std::string &base_filename, std::string &message, stats_t<T, ITYPE> &stats)
{
    std::string filename = base_filename + ".csv";
    int count = 1;

    struct stat buffer;
    while ( stat (filename.c_str(), &buffer) == 0 ) {
        filename = base_filename + "_" + std::to_string(count) + ".csv";
        count++;
    }

    std::ofstream csv_file;
    csv_file.open( filename );

    csv_file << message << "," << "\n";
    csv_file << stats.name << ",";
    for (ITYPE i = 0; i < stats.size; i++) {
        csv_file << stats.records[i] << ",";
    }
    csv_file << "\n";

    csv_file.close();

}

template<typename T, typename ITYPE>
void stats_arr_to_csv(std::string &base_filename, std::string &message, ITYPE num_stats_objects, stats_t<T, ITYPE> *stats, bool append = false)
{
    std::string filename = base_filename + ".csv";
    std::ofstream csv_file;

    if (!append) {
        int count = 1;
        struct stat buffer;
        while ( stat (filename.c_str(), &buffer) == 0 ) {
            filename = base_filename + "_" + std::to_string(count) + ".csv";
            count++;
        }
        csv_file.open( filename );
    } else {
        csv_file.open( filename, std::ios_base::app );
    }

    csv_file << message << "\n";

    for ( ITYPE id = 0; id < num_stats_objects; id++ ) {
        csv_file << stats[id].name << ",";

        for ( ITYPE i = 0; i < stats[id].size; i++ ) {
            csv_file << stats[id].records[i] << ",";
        }

        csv_file << "\n";
    }

    csv_file.close();

}


#endif
