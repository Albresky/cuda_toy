.PHONY: all prof clean

prof:
	@echo "Profiling GEMM..."
	nsys profile gemm.exe |& tee gemm_op0_profiling.log
	nsys stats  report1.nsys-rep |& tee -a gemm_op0_profiling.log

clean:
	rm -rf *.exe
	rm -rf *.log
	rm -rf *.nsys-rep *.sqlite