################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../cudasrc/EFP/mainForecastingCodes/test.cu 

CU_DEPS += \
./cudasrc/EFP/mainForecastingCodes/test.d 

OBJS += \
./cudasrc/EFP/mainForecastingCodes/test.o 


# Each subdirectory must supply rules for building sources it contributes
cudasrc/EFP/mainForecastingCodes/%.o: ../cudasrc/EFP/mainForecastingCodes/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-6.5/bin/nvcc -O3 -gencode arch=compute_20,code=sm_20  -odir "cudasrc/EFP/mainForecastingCodes" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.5/bin/nvcc -O3 --compile --relocatable-device-code=false -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


