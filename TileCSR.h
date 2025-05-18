/*
Author: Petrus E. Manurung
*/

#pragma once

#include "cx.h"

template<typename ValueType, int tileSize = 16>
struct __align__(16) TileCSR_rev {
    public:
    using MaskType  = uint16_t;
    using IdxType   = uint8_t;

    ValueType       *vals;              // 8 byte
    IdxType         *rowColIdx;         // 8 byte
    MaskType        mask[tileSize];     // 32 byte
    IdxType         rowPtr[tileSize];   // 16 byte
};

template<typename ValueType, int tileSize = 16>
struct __align__(16) TileCSR_C_rev {
    public:
    using MaskType = unsigned;
    using IdxType  = uint8_t;

    ValueType       *vals;              // 8
    IdxType         *rowColIdx;         // 8
    MaskType        *mask;
    // MaskType        mask[tileSize/2];   // 32
    // IdxType         rowPtr[tileSize];   // 16
};