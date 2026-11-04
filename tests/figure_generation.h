#ifndef NMPDE_FIGURE_GENERATION_H
#define NMPDE_FIGURE_GENERATION_H

#include <cstdlib>
#include <filesystem>
#include <string>

namespace CourseFigures
{
  inline bool
  needs_generation(const std::filesystem::path &path)
  {
    const char *force = std::getenv("NMPDE_REGENERATE_FIGURES");
    return (force != nullptr && std::string(force) == "1") ||
           !std::filesystem::is_regular_file(path) ||
           std::filesystem::file_size(path) == 0;
  }
} // namespace CourseFigures

#endif
