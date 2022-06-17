#pragma once

#include <string>
#include <vector>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <filesystem>

#include <nvrtc.h>

#define SAMPLES_ABSOLUTE_INCLUDE_DIRS \
  "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.4.0/include", \
  "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/include", 

// NVRTC compiler options
#define CUDA_NVRTC_OPTIONS  \
  "-std=c++11", \
  "-arch", \
  "compute_50", \
  "-use_fast_math", \
  "-lineinfo", \
  "-default-device", \
  "-rdc", \
  "true", \
  "-D__x86_64",

static bool readSourceFile(std::string& str, const std::string& filename)
{
    // Try to open file
    std::ifstream file(filename.c_str(), std::ios::binary);
    if (file.good())
    {
        // Found usable source file
        std::vector<unsigned char> buffer = std::vector<unsigned char>(std::istreambuf_iterator<char>(file), {});
        str.assign(buffer.begin(), buffer.end());
        return true;
    }
    return false;
}

static void getCuStringFromFile(std::string& cu, const char* filename)
{
    if (readSourceFile(cu, filename))
    {
        return;
    }

    // Wasn't able to find or open the requested file
    throw std::runtime_error("Couldn't open source file " + std::string(filename));
}

static std::string g_nvrtcLog;

static void getPtxFromCuString(std::string& ptx,
    const char* cu_source,
    const char* name,
    const char** log_string,
    const std::vector<const char*>& compiler_options)
{
    // Create program
    nvrtcProgram prog = 0;
    NVRTC_CHECK_ERROR(nvrtcCreateProgram(&prog, cu_source, name, 0, NULL, NULL));

    // Gather NVRTC options
    std::vector<const char*> options;

    // sample_dir = std::string("-I") + base_dir + '/' + sample_directory;
    // options.push_back(sample_dir.c_str());

    // Collect include dirs
    std::vector<std::string> include_dirs;
    const char* abs_dirs[] = { SAMPLES_ABSOLUTE_INCLUDE_DIRS };

    for (const char* dir : abs_dirs)
    {
        include_dirs.push_back(std::string("-I") + dir);
    }
    for (const std::string& dir : include_dirs)
    {
        options.push_back(dir.c_str());
    }

    // Collect NVRTC options
    std::copy(std::begin(compiler_options), std::end(compiler_options), std::back_inserter(options));

    // JIT compile CU to PTX
    const nvrtcResult compileRes = nvrtcCompileProgram(prog, (int)options.size(), options.data());

    // Retrieve log output
    size_t log_size = 0;
    NVRTC_CHECK_ERROR(nvrtcGetProgramLogSize(prog, &log_size));
    g_nvrtcLog.resize(log_size);
    if (log_size > 1)
    {
        NVRTC_CHECK_ERROR(nvrtcGetProgramLog(prog, &g_nvrtcLog[0]));
        if (log_string)
            *log_string = g_nvrtcLog.c_str();
    }
    if (compileRes != NVRTC_SUCCESS)
        throw std::runtime_error("NVRTC Compilation failed.\n" + g_nvrtcLog);

    // Retrieve PTX code
    size_t ptx_size = 0;
    NVRTC_CHECK_ERROR(nvrtcGetPTXSize(prog, &ptx_size));
    ptx.resize(ptx_size);
    NVRTC_CHECK_ERROR(nvrtcGetPTX(prog, &ptx[0]));

    // Cleanup
    NVRTC_CHECK_ERROR(nvrtcDestroyProgram(&prog));
}

const char* readPTX(size_t* size)
{
    std::string* ptx, cu;
    ptx = new std::string();

    const char** log = NULL;
    const std::vector<const char*> compilerOptions = { CUDA_NVRTC_OPTIONS };

    std::cout << std::filesystem::current_path() << std::endl;

    const char* filename = "./renderer.cu";
    getCuStringFromFile(cu, filename);
    getPtxFromCuString(*ptx, cu.c_str(), "renderer", log, compilerOptions);

    *size = ptx->size();
    return ptx->c_str();
}