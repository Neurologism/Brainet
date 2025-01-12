cmake_minimum_required(VERSION 3.28)
project(brainet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
include_directories(${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/json_interface)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")

# List of C++ source files
set(CPP_SOURCES
        src/variable.cpp
        src/model.cpp
        src/graph.cpp
        src/reader.cpp
        src/preprocessing/preprocessing.cpp
        src/module/dataset.cpp
        src/module/dense.cpp
        src/module/layer.cpp
        src/module/loss.cpp
        src/module/module.cpp
        src/operation/activation_function/activation_function.cpp
        src/operation/activation_function/heavyside_step.cpp
        src/operation/activation_function/hyperbolic_tangent.cpp
        src/operation/activation_function/linear.cpp
        src/operation/activation_function/parametric_relu.cpp
        src/operation/activation_function/rectified_linear_unit.cpp
        src/operation/activation_function/sigmoid.cpp
        src/operation/activation_function/softmax.cpp
        src/operation/loss_functions/error_rate.cpp
        src/operation/loss_functions/loss_function.cpp
        src/operation/parameter_norm_penalties/L1_Norm.cpp
        src/operation/parameter_norm_penalties/L2_Norm.cpp
        src/operation/parameter_norm_penalties/parameter_norm_penalty.cpp
        src/operation/processing/average.cpp
        src/operation/processing/dropout.cpp
        src/operation/processing/one_hot.cpp
        src/operation/processing/padding.cpp
        src/operation/surrogate_loss_functions/cross_entropy.cpp
        src/operation/surrogate_loss_functions/mean_absolute_error.cpp
        src/operation/surrogate_loss_functions/mse.cpp
        src/operation/weight_initialization/he_initialization.cpp
        src/operation/weight_initialization/normal_distribution_initializer.cpp
        src/operation/weight_initialization/normalized_initialization.cpp
        src/operation/weight_initialization/weight_initializer.cpp
        src/operation/weight_initialization/weight_matrix_initializer.cpp
        src/operation/weight_initialization/uniform_distribution_initializer.cpp
        src/operation/matmul.cpp
        src/operation/operation.cpp
        src/optimizer/sgd.cpp
        src/optimizer/adagrad.cpp
        src/optimizer/adam.cpp
        src/optimizer/momentum_sgd.cpp
        src/optimizer/nesterov_momentum.cpp
        src/optimizer/rmsprop.cpp
        src/optimizer/rmsprop_nesterov.cpp
        src/logger.cpp
        src/datatypes/matrix.cpp
        src/datatypes/tensor.cpp
        src/datatypes/vector.cpp
)

# List of CUDA source files
#set(CUDA_SOURCES
#        src/operation/matmul.cu
#)

# Create a library target for C++ sources
add_library(brainet_cpp STATIC ${CPP_SOURCES})

# Create a library target for CUDA sources
#add_library(brainet_cuda STATIC ${CUDA_SOURCES})

# Link the CUDA library with the C++ library
#target_link_libraries(brainet_cpp brainet_cuda)

# Create executables
add_executable(example tests/example.cpp)
add_executable(json json_interface/run_json.cpp)

# Link the executables with the C++ library (which is already linked with the CUDA library)
target_link_libraries(example brainet_cpp)
target_link_libraries(json brainet_cpp)