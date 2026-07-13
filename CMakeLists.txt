cmake_minimum_required(VERSION 3.15)
project(mllm_cpp_framework CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/W4 /O2 /arch:AVX2 /fp:fast /openmp)
else()
    add_compile_options(-Wall -Wextra -O2 -march=native -ffast-math -fopenmp)
    add_link_options(-fopenmp)
endif()

# Engine sources shared by the CLI and the web server (everything but the
# per-executable main).
set(ENGINE_SOURCES
    src/memory_budget.cpp
    src/inference_arenas.cpp
    src/gguf_loader.cpp
    src/config_loader.cpp
    src/safetensors_loader.cpp
    src/forward_pass.cpp
    src/qwen_forward.cpp
    src/qwen_model.cpp
    src/tokenizer.cpp
)

# CLI: text in -> generated text out.
add_executable(mllm_engine src/main.cpp ${ENGINE_SOURCES})
target_include_directories(mllm_engine PRIVATE include)

# Web server: browser test UI for the text-to-text engine (Windows/Winsock).
if(WIN32)
    add_executable(mllm_server src/server.cpp ${ENGINE_SOURCES})
    target_include_directories(mllm_server PRIVATE include)
    target_link_libraries(mllm_server ws2_32)
endif()
