#pragma once
#include <array>
namespace qc::demo {
constexpr int station_frames = 35;
constexpr int inspect_begin = 10;
constexpr int inspect_end = 22;
constexpr int source_fps = 18;
constexpr std::array<std::array<int, 4>, 15> part_rois = {{
    {0, 0, 640, 640},
    {0, 0, 640, 640},
    {0, 0, 640, 640},
    {0, 0, 640, 640},
    {0, 0, 640, 640},
    {0, 164, 640, 312},
    {0, 164, 640, 312},
    {0, 164, 640, 312},
    {0, 0, 640, 640},
    {0, 0, 640, 640},
    {0, 0, 640, 640},
    {0, 0, 640, 640},
    {0, 0, 640, 640},
    {0, 0, 640, 640},
    {0, 0, 640, 640},
}};
}
