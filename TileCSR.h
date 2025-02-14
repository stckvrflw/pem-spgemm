/*
Author: Petrus E. Manurung
*/

#pragma once

#include "cx.h"

// ValueType = unsigned; size = 1328 bytes; max 49152 bytes -> deviceQuery
// notice every member is 16 bytes aligned
template<typename ValueType, int tileSize = 16>
struct __align__(16) TileCSR {
    public:
    using MaskType  = uint16_t;
    using IdxType   = uint8_t;

    ValueType       vals[tileSize * tileSize];
    MaskType        mask[tileSize];
    IdxType         rowPtr[tileSize];               
    IdxType         rowColIdx[tileSize * tileSize];
};

template<typename ValueType, int tileSize = 16>
struct __align__(16) TileCSR_C {
    public:
    using MaskType  = uint16_t;
    using IdxType   = uint8_t;

    ValueType       vals[tileSize * tileSize];
    MaskType        mask[tileSize];
};