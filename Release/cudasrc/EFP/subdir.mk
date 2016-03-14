################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../cudasrc/EFP/Representation.cpp \
../cudasrc/EFP/Util.cpp \
../cudasrc/EFP/autocorr.cpp \
../cudasrc/EFP/lregress.cpp 

CU_SRCS += \
../cudasrc/EFP/CUDAEval.cu 

CU_DEPS += \
./cudasrc/EFP/CUDAEval.d 

OBJS += \
./cudasrc/EFP/CUDAEval.o \
./cudasrc/EFP/Representation.o \
./cudasrc/EFP/Util.o \
./cudasrc/EFP/autocorr.o \
./cudasrc/EFP/lregress.o 

CPP_DEPS += \
./cudasrc/EFP/Representation.d \
./cudasrc/EFP/Util.d \
./cudasrc/EFP/autocorr.d \
./cudasrc/EFP/lregress.d 


# Each subdirectory must supply rules for building sources it contributes
cudasrc/EFP/%.o: ../cudasrc/EFP/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-6.5/bin/nvcc -O3 -gencode arch=compute_20,code=sm_20  -odir "cudasrc/EFP" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.5/bin/nvcc -O3 --compile --relocatable-device-code=false -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

cudasrc/EFP/%.o: ../cudasrc/EFP/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-6.5/bin/nvcc -O3 -gencode arch=compute_20,code=sm_20  -odir "cudasrc/EFP" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.5/bin/nvcc -O3 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


