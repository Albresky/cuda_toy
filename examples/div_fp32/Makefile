NVCC ?= /usr/local/cuda-13.0/bin/nvcc
CUOBJDUMP ?= /usr/local/cuda-13.0/bin/cuobjdump
NVDISASM ?= /usr/local/cuda-13.0/bin/nvdisasm
SM_CODE := sm_100a
ARCH := -arch=$(SM_CODE)
SRC := div_fp32.cu
BIN := div_fp32_$(SM_CODE)
PTX := $(BIN).ptx
CUBIN := $(BIN).cubin
SASS := $(BIN).sass
ASM := $(BIN).asm
OCFG_PNG := $(BIN)_ocfg.png
OBBCFG_PNG := $(BIN)_obb_cfg.png

.PHONY: all build clean ptx cubin sass

all: build

build: $(BIN)

$(BIN): $(SRC)
	$(NVCC) $(ARCH) -O3 -lineinfo -o $@ $<

ptx: $(PTX)

$(PTX): $(SRC)
	$(NVCC) $(ARCH) -ptx -o $@ $<

cubin: $(CUBIN)

$(CUBIN): $(SRC)
	$(NVCC) $(ARCH) -cubin -o $@ $<

sass: $(SASS)

$(SASS): $(CUBIN)
	$(CUOBJDUMP) --dump-sass $< > $@

cfg: $(CUBIN)
	$(CUOBJDUMP) --dump-ocfg --output-file=$(OCFG_PNG) $<
	$(CUOBJDUMP) --dump-obb-cfg --output-file=$(OBBCFG_PNG) $<

asm: $(ASM)

$(ASM): $(CUBIN)
	$(NVDISASM) -gi $< > $@

run: $(BIN)
	./$(BIN)

clean:
	rm -f $(BIN) $(PTX) $(CUBIN) $(SASS)
