BUILD_DIR=build
include $(N64_INST)/include/n64.mk

OBJS = $(BUILD_DIR)/ulcdemo.o $(BUILD_DIR)/ulc.o

assets_wav = $(wildcard assets/*.wav)
assets_ulc = $(wildcard assets/*.ulc)

assets_conv = $(addprefix filesystem/,$(notdir $(assets_wav:%.wav=%.wav64))) \
              $(addprefix filesystem/,$(notdir $(assets_ulc:%.ulc=%.ulc)))

all: ulcdemo.z64

# Run audioconv on all WAV files under assets/
# We do this file by file, but we could even do it just once for the whole
# directory, because audioconv64 supports directory walking.
filesystem/%.wav64: assets/%.wav
	@mkdir -p $(dir $@)
	@echo "    [AUDIO] $@"
	@$(N64_AUDIOCONV) --wav-resample 22050 --wav-compress vadpcm,huffman=false -o filesystem $<

filesystem/%.ulc: assets/%.ulc
	@mkdir -p $(@D)
	cp $< $@


$(BUILD_DIR)/ulcdemo.dfs: $(assets_conv)
$(BUILD_DIR)/ulcdemo.elf: $(OBJS)

ulcdemo.z64: N64_ROM_TITLE="Mixer Test"
ulcdemo.z64: $(BUILD_DIR)/ulcdemo.dfs

clean:
	rm -f $(BUILD_DIR)/* ulcdemo.z64 $(assets_conv)

-include $(wildcard $(BUILD_DIR)/*.d)

.PHONY: all clean
