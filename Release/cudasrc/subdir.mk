################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../cudasrc/mainEFP.cpp 

OBJS += \
./cudasrc/mainEFP.o 

CPP_DEPS += \
./cudasrc/mainEFP.d 


# Each subdirectory must supply rules for building sources it contributes
cudasrc/%.o: ../cudasrc/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-6.5/bin/nvcc -O3 -gencode arch=compute_20,code=sm_20  -odir "cudasrc" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.5/bin/nvcc -O3 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


