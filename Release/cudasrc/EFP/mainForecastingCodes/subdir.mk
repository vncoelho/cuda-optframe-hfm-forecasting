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
	/usr/local/cuda-8.0/bin/nvcc -O3 -gencode arch=compute_30,code=sm_30  -odir "cudasrc/EFP/mainForecastingCodes" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -O3 --compile --relocatable-device-code=false -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


