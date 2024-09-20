cmake_minimum_required(VERSION 3.29)
project(brainet)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src, ${CMAKE_SOURCE_DIR}/json_interface)

add_executable(brainet
        include/datatypes/tensor.hpp
        json_interface/run_json.cpp
        src/variable.cpp
        src/model.cpp
        include/model.hpp
        include/variable.hpp
        src/graph.cpp
        include/graph.hpp
        include/dependencies.hpp
        src/reader.cpp
        include/reader.hpp
        include/datatypes/matrix.hpp
        src/preprocessing/preprocessing.cpp
        include/preprocessing/preprocessing.hpp
        src/module/dataset.cpp
        src/module/dense.cpp
        src/module/layer.cpp
        src/module/loss.cpp
        src/module/module.cpp
        include/module/dataset.hpp
        include/module/module.hpp
        include/module/dense.hpp
        include/module/layer.hpp
        include/module/loss.hpp
        src/operation/activation_function/activation_function.cpp
        src/operation/activation_function/heavyside_step.cpp
        src/operation/activation_function/hyperbolic_tangent.cpp
        src/operation/activation_function/linear.cpp
        src/operation/activation_function/parametric_relu.cpp
        src/operation/activation_function/rectified_linear_unit.cpp
        src/operation/activation_function/sigmoid.cpp
        src/operation/activation_function/softmax.cpp
        src/operation/loss_functions/error_rate.cpp
        include/operation/loss_functions/error_rate.hpp
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
        src/operation/weight_initialization/uniform_distribution_initializer.hpp.cpp
        src/operation/weight_initialization/weight_initializer.cpp
        src/operation/weight_initialization/weight_matrix_initializer.cpp
        src/operation/matmul.cpp
        src/operation/operation.cpp
        src/optimizer/sgd.cpp
        src/optimizer/adagrad.cpp
        src/optimizer/adam.cpp
        src/optimizer/momentum_sgd.cpp
        src/optimizer/nesterov_momentum.cpp
        src/optimizer/rmsprop.cpp
        src/optimizer/rmsprop_nesterov.cpp
        include/brainet.hpp
        include/module/module_variant.hpp
        include/operation/activation_function/activation_function_variant.hpp
        include/operation/parameter_norm_penalties/norm_variant.hpp
        include/operation/loss_functions/loss_variant.hpp
        include/optimizer/optimizer_variant.hpp
)