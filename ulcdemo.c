#include <libdragon.h>
#include "ulc.h"

static inline uint16_t bswap16(uint16_t x) {
    return (x >> 8) | (x << 8);
}

static inline uint32_t bswap32(uint32_t x) {
    return  ((x >> 24) & 0x000000FF) |
            ((x >> 8)  & 0x0000FF00) |
            ((x << 8)  & 0x00FF0000) |
            ((x << 24) & 0xFF000000);
}

static inline int clamp_s16(int v) {
	if ((unsigned int)(v + 32768) > 65535) {
		if (v < -32768) { return -32768; }
		if (v >  32767) { return  32767; }
	}
	return v;
}

int main(void) {
    timer_init();
    debug_init_isviewer();
    audio_init(44100, 4);
    dfs_init(DFS_DEFAULT_LOCATION);
    console_init();
    console_set_render_mode(RENDER_MANUAL);
    console_set_debug(false);

    char *path = "rom://demo-32.ulc";
	
	
	FILE *fh = fopen(path, "rb");
	assertf(fh, "ERROR: Unable to open input file (%s).\n", path);

	// Read header
	struct FileHeader_t header;
	uint32_t header_read = fread(&header, sizeof(header), 1, fh);
	assertf(header_read == 1, "ERROR: Input file is not a valid ULC container.\n");

	// meh
	header.Magic        = bswap32(header.Magic);
	header.BlockSize    = bswap16(header.BlockSize);
	header.MaxBlockSize = bswap16(header.MaxBlockSize);
	header.nBlocks      = bswap32(header.nBlocks);
	header.RateHz       = bswap32(header.RateHz);
	header.nChan        = bswap16(header.nChan);
	header.RateKbps     = bswap16(header.RateKbps);
	header.StreamOffs   = bswap32(header.StreamOffs);

	assertf(header.Magic == HEADER_MAGIC, "ERROR: Input file is not a valid ULC container.\n");


	// Define the stream buffer size
	int read_buf_len = (16 * 1024);
	if((int)header.MaxBlockSize > read_buf_len) {
		read_buf_len = header.MaxBlockSize;
	}

	// Allocate decoding buffer and stream buffer
	uint8_t *buf = malloc(BUFFER_ALIGNMENT-1 + sizeof(float) * 2 * header.BlockSize * header.nChan + read_buf_len);
	assertf(buf, "ERROR: Couldn't allocate decoding buffer.\n");
	float *decode_buf = (float  *)(buf + (-(uintptr_t)buf % BUFFER_ALIGNMENT));
	uint8_t *read_buf = (uint8_t*)(decode_buf + header.BlockSize*header.nChan);

	// Create decoder
	struct ULC_DecoderState_t decoder = {.nChan = header.nChan, .BlockSize = header.BlockSize};
	int decoder_init = ULC_DecoderState_Init(&decoder);
	assertf(decoder_init > 0, "ERROR: Unable to initialize decoder.\n");


	fseek(fh, header.StreamOffs, SEEK_SET);
	fread(read_buf, read_buf_len, 1, fh);

	int out_len = header.BlockSize * header.nChan;
	int16_t *out_buff = malloc(out_len * sizeof(int16_t));

	for (uint32_t block = 0; block < header.nBlocks; block++) {
		
		// Decode block
		uint64_t decode_start = get_ticks_ms();
		int decode_len = (ULC_DecodeBlock(&decoder, decode_buf, read_buf) + 7) / 8u;
		assertf(decode_len, "ERROR: Corrupted stream.\n");

		// Slide read buffer
		memcpy(read_buf, read_buf+decode_len, read_buf_len-decode_len);
		fread(read_buf + read_buf_len-decode_len, decode_len, 1, fh);

		// Convert float to s16
		for (int i = 0; i < out_len; i++) {
			out_buff[i] = clamp_s16(decode_buf[i] * 32767);
		}
		uint64_t decode_time = get_ticks_ms() - decode_start;
			

		// Write output
		uint64_t push_start = get_ticks_ms();
		audio_push(out_buff, header.BlockSize, true);
		uint64_t push_time = get_ticks_ms() - push_start;

		console_clear();
		printf("Playing %s: %d channels, %ld hz, %d kbps\n", path, header.nChan, header.RateHz, header.RateKbps);
		printf("decode: %2lld ms, wait: %2lld\n", decode_time, push_time);
		console_render();
	}
}