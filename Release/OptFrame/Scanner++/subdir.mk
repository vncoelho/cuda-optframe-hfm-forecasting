################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../OptFrame/Scanner++/Scanner.cpp 

OBJS += \
./OptFrame/Scanner++/Scanner.o 

CPP_DEPS += \
./OptFrame/Scanner++/Scanner.d 


# Each subdirectory must supply rules for building sources it contributes
OptFrame/Scanner++/%.o: ../OptFrame/Scanner++/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-6.5/bin/nvcc -O3 -gencode arch=compute_20,code=sm_20  -odir "OptFrame/Scanner++" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.5/bin/nvcc -O3 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


