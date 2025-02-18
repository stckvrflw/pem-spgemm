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

    ValueType       vals[tileSize * tileSize];      // 1024 byte -> float
    MaskType        mask[tileSize];                 // 32 byte
    IdxType         rowPtr[tileSize];               // 16 byte
    IdxType         rowColIdx[tileSize * tileSize]; // 256 byte
};

template<typename ValueType, int tileSize = 16>
struct __align__(16) TileCSR_rev {
    public:
    using MaskType  = uint16_t;
    using IdxType   = uint8_t;

    ValueType       *vals;              // 8 byte
    IdxType         *rowColIdx;          // 8 byte
    MaskType        mask[tileSize];     // 32 byte
    IdxType         rowPtr[tileSize];   // 16 byte
};

template<typename ValueType, int tileSize = 16>
struct __align__(16) TileCSR_C {
    public:
    using MaskType  = uint16_t;
    using IdxType   = uint8_t;

    ValueType       vals[tileSize * tileSize];
    MaskType        mask[tileSize];
};