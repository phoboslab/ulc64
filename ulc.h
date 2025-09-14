#ifndef ULC_H
#define ULC_H

/**************************************/
#include <stdint.h>
/**************************************/
#define BUFFER_ALIGNMENT 64u //! __mm512
/**************************************/

//! File header
#define HEADER_MAGIC (uint32_t)('U' | 'L'<<8 | 'C'<<16 | '2'<<24)
struct FileHeader_t {
	uint32_t Magic;        //! [00h] Magic value/signature
	uint16_t BlockSize;    //! [04h] Transform block size
	uint16_t MaxBlockSize; //! [06h] Largest block size (in bytes; 0 = Unknown)
	uint32_t nBlocks;      //! [08h] Number of blocks
	uint32_t RateHz;       //! [0Ch] Playback rate
	uint16_t nChan;        //! [10h] Channels in stream
	uint16_t RateKbps;     //! [12h] Nominal coding rate
	uint32_t StreamOffs;   //! [14h] Offset of data stream
};


#include <math.h>
#include <stdint.h>
#include <string.h>
/**************************************/
// #include "ulcEncoder.h"
/**************************************/
#define ABS(x) ((x) < 0 ? (-(x)) : (x))
#define SQR(x) ((x)*(x))
/**************************************/
#define ULC_FORCED_INLINE static inline __attribute__((always_inline))
/**************************************/

//! Subblock decimation pattern
//! Each subblock is coded in 4 bits (LSB to MSB):
//!  Bit0..2: Subblock shift (ie. BlockSize >> Shift)
//!  Bit3:    Transient flag (ie. apply overlap scaling to that subblock)
typedef uint_least16_t ULC_SubBlockDecimationPattern_t;
ULC_FORCED_INLINE
ULC_SubBlockDecimationPattern_t ULCi_SubBlockDecimationPattern(int WindowCtrl) {
	static const ULC_SubBlockDecimationPattern_t Pattern[] = {
		0x0000 | 0x0000, //! 0000: N/1 (Unused)
		0x0000 | 0x0008, //! 0001: N/1*
		0x0011 | 0x0008, //! 0010: N/2*,N/2
		0x0011 | 0x0080, //! 0011: N/2,N/2*
		0x0122 | 0x0008, //! 0100: N/4*,N/4,N/2
		0x0122 | 0x0080, //! 0101: N/4,N/4*,N/2
		0x0221 | 0x0080, //! 0110: N/2,N/4*,N/4
		0x0221 | 0x0800, //! 0111: N/2,N/4,N/4*
		0x1233 | 0x0008, //! 1000: N/8*,N/8,N/4,N/2
		0x1233 | 0x0080, //! 1001: N/8,N/8*,N/4,N/2
		0x1332 | 0x0080, //! 1010: N/4,N/8*,N/8,N/2
		0x1332 | 0x0800, //! 1011: N/4,N/8,N/8*,N/2
		0x2331 | 0x0080, //! 1100: N/2,N/8*,N/8,N/4
		0x2331 | 0x0800, //! 1101: N/2,N/8,N/8*,N/4
		0x3321 | 0x0800, //! 1110: N/2,N/4,N/8*,N/8
		0x3321 | 0x8000, //! 1111: N/2,N/4,N/8,N/8*
	};
	return Pattern[WindowCtrl >> 4];
}


//! Decoder state structure
//! NOTE:
//!  -The global state data must be set before calling ULC_DecoderState_Init()
//!  -{nChan, BlockSize,} must not change after calling ULC_EncoderState_Init()
struct ULC_DecoderState_t {
	//! Global state (do not change after initialization)
	int nChan;     //! Channels in encoding scheme
	int BlockSize; //! Transform block size

	//! Decoding state
	//! Buffer memory layout:
	//!  Data:
	//!   char  _Padding[];
	//!   float TransformBuffer[BlockSize]
	//!   float TransformTemp  [nChan * BlockSize]
	//!   float TransformInvLap[nChan * BlockSize/2]
	//! BufferData contains the pointer returned by malloc()
	//! TransformTemp[] is large because we need to interleave the output.
	int    LastSubBlockSize; //! Size of last [sub]block processed
	void  *BufferData;
	float *TransformBuffer;
	float *TransformTemp;
	float *TransformInvLap;
};

/**************************************/

//! Initialize decoder state
//! On success, returns a non-negative value
//! On failure, returns a negative value
int ULC_DecoderState_Init(struct ULC_DecoderState_t *State);

//! Destroy decoder state
void ULC_DecoderState_Destroy(struct ULC_DecoderState_t *State);

/**************************************/

//! Decode block
//! NOTE:
//!  -Output data will have its channels arranged sequentially;
//!   For example:
//!   {
//!    0,1,2,3...BlockSize-1, //! Chan0
//!    0,1,2,3...BlockSize-1, //! Chan1
//!   }
//!  -SrcBuffer will only be accessed via bytes.
//! Returns the number of bits read.
int ULC_DecodeBlock(struct ULC_DecoderState_t *State, float *DstData, const void *SrcBuffer);

/**************************************/
//! EOF
/**************************************/


#endif