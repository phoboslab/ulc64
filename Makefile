BUILD_DIR=build
include $(N64_INST)/include/n64.mk

OBJS = $(BUILD_DIR)/ulcdemo.o $(BUILD_DIR)/ulc.o

assets_ulc = $(wildcard assets/*.ulc)
assets_conv = $(addprefix filesystem/,$(notdir $(assets_ulc:%.ulc=%.ulc)))

all: ulcdemo.z64

filesystem/%.ulc: assets/%.ulc
	@mkdir -p $(@D)
	cp $< $@


$(BUILD_DIR)/ulcdemo.dfs: $(assets_conv)
$(BUILD_DIR)/ulcdemo.elf: $(OBJS)

ulcdemo.z64: N64_ROM_TITLE="ULC64"
ulcdemo.z64: $(BUILD_DIR)/ulcdemo.dfs

clean:
	rm -f $(BUILD_DIR)/* ulcdemo.z64 $(assets_conv)

-include $(wildcard $(BUILD_DIR)/*.d)

.PHONY: all clean
