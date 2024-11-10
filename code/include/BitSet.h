#ifndef BITSET_H
#define BITSET_H

#include <cassert>
#include <cstring>
// #include <bitset>
// #include <bit>
#include <set>

#include "utils/util.h"

#define BITS_IN_BYTE        8
#define BITS_IN_SIZE_T      ( sizeof(size_t) * BITS_IN_BYTE )
#define BITSET_INDEX(n)     ( n / BITS_IN_SIZE_T )
#define BIT_INDEX(n)        ( n & (BITS_IN_SIZE_T - 1) )
#define MAX_SIZE            8

// #define DEBUG_BITSET_CREATION

template<typename ITYPE = int64_t>
class BitSet {

    public:
        ITYPE num_entries;
        ITYPE data_store_size;
        // size_t *bitset;
        size_t bitset[MAX_SIZE];

        #ifdef DEBUG_BITSET
            std::set<ITYPE> active_entries;
        #endif

    public:
        BitSet(ITYPE num_entries);
        BitSet() { }
        ~BitSet() {
            // if (bitset != nullptr) { delete[] bitset;
            //     bitset = nullptr;
            // }
        }
        BitSet( const BitSet &rhs ) : num_entries(rhs.num_entries), data_store_size(rhs.data_store_size) {
            // this->bitset = new size_t[data_store_size]();
        }

        void init(ITYPE num_entries);
        inline void init(ITYPE num_entries, ITYPE data_store_size);

        void set(ITYPE n);  // set bit n to true
        bool get(ITYPE n);  // get bit n
        void reset();       // reset all bits
        ITYPE count();      // return bits set to true
        ITYPE or_count(BitSet &rhs);  // return bits set in either (i.e. set union)

        // perform an 'or' operation on the current bitset with the rhs
        void op_or(BitSet &rhs);

        BitSet& operator |= (BitSet const &rhs);
        BitSet& operator &= (BitSet const &rhs);
        BitSet& operator ^= (BitSet const &rhs);
        BitSet& operator = (BitSet const &rhs);
        BitSet& operator -= (BitSet const &rhs);
        BitSet& operator *= (BitSet const &rhs);


        BitSet operator | (BitSet const &rhs);
        BitSet operator & (BitSet const &rhs);
        BitSet operator ^ (BitSet const &rhs);
        BitSet operator - (BitSet const &rhs);
        BitSet operator + (BitSet const &rhs);
        BitSet operator * (BitSet const &rhs);
        BitSet operator ~ ();

        bool operator != (BitSet const &rhs);
};

template<typename ITYPE>
BitSet<ITYPE>::BitSet(ITYPE num_entries) : num_entries(num_entries)
{
    this->data_store_size = CEIL( num_entries, (sizeof(size_t) * BITS_IN_BYTE) );

    #ifdef DEBUG_BITSET_CREATION
        std::cerr << "Allocating bitset width: " << num_entries << " using " << this->data_store_size << " size_t entries" << std::endl;
    #endif

    for ( ITYPE i = 0; i < data_store_size; i++ ) {
        bitset[i] = 0;
    }
    // this->bitset = new size_t[data_store_size]();
    // assert(this->bitset && "could not allocate memory for bitset");
}

template<typename ITYPE>
inline void BitSet<ITYPE>::init(ITYPE num_entries)
{
    this->num_entries = num_entries;
    this->data_store_size = CEIL( num_entries, (sizeof(size_t) * BITS_IN_BYTE) );
    std::memset( this->bitset, 0, data_store_size * sizeof(size_t) );
    // Allocate memory only if necessary -- otherwise just reset to zero

    /*
    if (this->bitset == nullptr) {
        this->bitset = new size_t[data_store_size]();
    } else {
        std::memset(this->bitset, 0, data_store_size * sizeof(size_t));
    }
    */

}

template <typename ITYPE>
inline void BitSet<ITYPE>::init(ITYPE num_entries, ITYPE data_store_size)
{
    this->num_entries = num_entries;
    this->data_store_size = data_store_size;
    std::memset( this->bitset, 0, data_store_size * sizeof(size_t) );
}

template<typename ITYPE>
void BitSet<ITYPE>::set(ITYPE n)
{
    ITYPE index = BITSET_INDEX(n);
    bitset[index] |=  ((size_t) 1) << BIT_INDEX(n) ;

    #ifdef DEBUG_BITSET
        active_entries.insert(n);
    #endif
}

template<typename ITYPE>
bool BitSet<ITYPE>::get(ITYPE n)
{
    ITYPE index = BITSET_INDEX(n);
    return (bitset[index] >> BIT_INDEX(n)) & 1;
}

template<typename ITYPE>
void BitSet<ITYPE>::reset()
{
    std::memset(this->bitset, 0, data_store_size * sizeof(size_t));
}

template<typename ITYPE>
ITYPE BitSet<ITYPE>::count()
{
    ITYPE count = 0;
    for (ITYPE i = 0; i < data_store_size; i++) {
        count += __builtin_popcountll(bitset[i]);
        // count += _mm_popcnt_u64(bitset[i]);
    }
    return count;
}

template<typename ITYPE>
void BitSet<ITYPE>::op_or(BitSet<ITYPE> &rhs)
{
    assert(num_entries != rhs.num_entries && "Bitset or operation on different size bitsets");

    for (ITYPE i = 0; i < data_store_size; i++) {
        bitset[i] |= rhs.bitset[i];
    }
}

template<typename ITYPE>
BitSet<ITYPE>& BitSet<ITYPE>::operator |= (BitSet<ITYPE> const &rhs)
{
    assert(this->num_entries == rhs.num_entries && "Bitset or operation on different size bitsets");

    for (ITYPE i = 0; i < data_store_size; i++) {
        bitset[i] |= rhs.bitset[i];
    }

    return *this;
}

template<typename ITYPE>
BitSet<ITYPE>& BitSet<ITYPE>::operator &= (BitSet<ITYPE> const &rhs)
{
    assert(this->num_entries == rhs.num_entries && "Bitset or operation on different size bitsets");

    for (ITYPE i = 0; i < data_store_size; i++) {
        bitset[i] &= rhs.bitset[i];
    }

    return *this;
}

template<typename ITYPE>
BitSet<ITYPE>& BitSet<ITYPE>::operator ^= (BitSet<ITYPE> const &rhs)
{
    assert(this->num_entries == rhs.num_entries && "Bitset or operation on different size bitsets");

    for (ITYPE i = 0; i < data_store_size; i++) {
        bitset[i] ^= rhs.bitset[i];
    }

    return *this;
}

template<typename ITYPE>
BitSet<ITYPE>& BitSet<ITYPE>::operator -= (BitSet<ITYPE> const &rhs)
{
    assert(this->num_entries == rhs.num_entries && "Bitset - operation on different size bitsets");

    for (ITYPE i = 0; i < data_store_size; i++) {
        bitset[i] &= ~rhs.bitset[i];
    }

    return *this;
}

template <typename ITYPE>
BitSet<ITYPE>& BitSet<ITYPE>::operator *= (BitSet<ITYPE> const &rhs)
{
    for (ITYPE i = 0; i < data_store_size; i++) {
        bitset[i] &= rhs.bitset[i];
    }

    return *this;
}

template<typename ITYPE>
BitSet<ITYPE>& BitSet<ITYPE>::operator = (BitSet<ITYPE> const &rhs)
{
    if (this->num_entries != rhs.num_entries) {
        print_error_exit("Bitset assignment on different size bitsets");
    }

    if (this->data_store_size == 1) {
        this->bitset[0] = rhs.bitset[0];
    } else if (this->data_store_size == 2) {
        this->bitset[0] = rhs.bitset[0];
        this->bitset[1] = rhs.bitset[1];
    } else if (this->data_store_size > 2) {
        std::memcpy(bitset, rhs.bitset, sizeof(*bitset) * data_store_size);
    }

    return *this;
}

template<typename ITYPE>
BitSet<ITYPE> BitSet<ITYPE>::operator | (BitSet<ITYPE> const &rhs)
{
    BitSet<ITYPE> temp(num_entries);
    for (ITYPE i = 0; i < data_store_size; i++) {
        temp.bitset[i] = bitset[i] | rhs.bitset[i];
    }

    return temp;
}

template<typename ITYPE>
BitSet<ITYPE> BitSet<ITYPE>::operator & (BitSet<ITYPE> const &rhs)
{
    BitSet<ITYPE> temp(num_entries);
    for (ITYPE i = 0; i < data_store_size; i++) {
        temp.bitset[i] = bitset[i] & rhs.bitset[i];
    }

    return temp;
}

template<typename ITYPE>
BitSet<ITYPE> BitSet<ITYPE>::operator ^ (BitSet<ITYPE> const &rhs)
{
    BitSet<ITYPE> temp(num_entries);
    for (ITYPE i = 0; i < data_store_size; i++) {
        temp.bitset[i] = bitset[i] ^ rhs.bitset[i];
    }

    return temp;
}

template <typename ITYPE>
BitSet<ITYPE> BitSet<ITYPE>::operator - (BitSet<ITYPE> const &rhs)
{
    BitSet<ITYPE> temp(num_entries);
    for (ITYPE i = 0; i < data_store_size; i++) {
        temp.bitset[i] = bitset[i] & (~rhs.bitset[i]);
    }

    return temp;
}

template <typename ITYPE>
BitSet<ITYPE> BitSet<ITYPE>::operator + (BitSet<ITYPE> const &rhs)
{
    BitSet<ITYPE> temp(num_entries);
    for (ITYPE i = 0; i < data_store_size; i++) {
        temp.bitset[i] = bitset[i] | rhs.bitset[i];
    }

    return temp;
}

template <typename ITYPE>
BitSet<ITYPE> BitSet<ITYPE>::operator * (BitSet<ITYPE> const &rhs)
{
    BitSet<ITYPE> temp(num_entries);
    for (ITYPE i = 0; i < data_store_size; i++) {
        temp.bitset[i] = bitset[i] & rhs.bitset[i];
    }

    return temp;
}

template <typename ITYPE>
BitSet<ITYPE> BitSet<ITYPE>::operator ~ ()
{
    BitSet<ITYPE> temp(num_entries);
    for (ITYPE i = 0; i < data_store_size; i++) {
        temp.bitset[i] = ~bitset[i];
    }

    return temp;
}

template<typename ITYPE>
bool BitSet<ITYPE>::operator != (BitSet<ITYPE> const &rhs)
{
    for(ITYPE i = 0; i < this->data_store_size; i++) {
        if (this->bitset[i] != rhs.bitset[i]) {
            return true;
        }
    }
    return false;
}

#endif // BITSET_H
