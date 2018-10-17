// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "Rand.h"
#include "Raw.h"

#include <iostream>

using namespace std;

// ----------------------------------------------------------------------------
// Initialize an Eigen::Vector2i vector.
// Output range: [vmin:vmax].
// ----------------------------------------------------------------------------
template <>
void UnitTest::Rand(
    vector<Eigen::Vector2i> &v,
    const Eigen::Vector2i &vmin,
    const Eigen::Vector2i &vmax,
    const int& seed)
{
    UnitTest::Raw raw(seed);

    Eigen::Vector2d factor;
    factor(0, 0) = (double)(vmax(0, 0) - vmin(0, 0)) / UnitTest::Raw::VMAX;
    factor(1, 0) = (double)(vmax(1, 0) - vmin(1, 0)) / UnitTest::Raw::VMAX;

    for (size_t i = 0; i < v.size(); i++)
    {
        v[i](0, 0) = vmin(0, 0) + (int)(raw.Next<int>() * factor(0, 0));
        v[i](1, 0) = vmin(1, 0) + (int)(raw.Next<int>() * factor(1, 0));
    }
}

// ----------------------------------------------------------------------------
// Initialize an Eigen::Vector3i vector.
// Output range: [vmin:vmax].
// ----------------------------------------------------------------------------
template <>
void UnitTest::Rand(
    vector<Eigen::Vector3i> &v,
    const Eigen::Vector3i &vmin,
    const Eigen::Vector3i &vmax,
    const int& seed)
{
    UnitTest::Raw raw(seed);

    Eigen::Vector3d factor;
    factor(0, 0) = (double)(vmax(0, 0) - vmin(0, 0)) / UnitTest::Raw::VMAX;
    factor(1, 0) = (double)(vmax(1, 0) - vmin(1, 0)) / UnitTest::Raw::VMAX;
    factor(2, 0) = (double)(vmax(2, 0) - vmin(2, 0)) / UnitTest::Raw::VMAX;

    for (size_t i = 0; i < v.size(); i++)
    {
        v[i](0, 0) = vmin(0, 0) + (int)(raw.Next<int>() * factor(0, 0));
        v[i](1, 0) = vmin(1, 0) + (int)(raw.Next<int>() * factor(1, 0));
        v[i](2, 0) = vmin(2, 0) + (int)(raw.Next<int>() * factor(2, 0));
    }
}

// ----------------------------------------------------------------------------
// Initialize an Eigen::Vector3d vector.
// Output range: [vmin:vmax].
// ----------------------------------------------------------------------------
template <>
void UnitTest::Rand(
    vector<Eigen::Vector3d> &v,
    const Eigen::Vector3d &vmin,
    const Eigen::Vector3d &vmax,
    const int& seed)
{
    UnitTest::Raw raw(seed);

    Eigen::Vector3d factor;
    factor(0, 0) = vmax(0, 0) - vmin(0, 0);
    factor(1, 0) = vmax(1, 0) - vmin(1, 0);
    factor(2, 0) = vmax(2, 0) - vmin(2, 0);

    for (size_t i = 0; i < v.size(); i++)
    {
        v[i](0, 0) = vmin(0, 0) + raw.Next<double>() * factor(0, 0);
        v[i](1, 0) = vmin(1, 0) + raw.Next<double>() * factor(1, 0);
        v[i](2, 0) = vmin(2, 0) + raw.Next<double>() * factor(2, 0);
    }
}

// ----------------------------------------------------------------------------
// Initialize a uint8_t vector.
// Output range: [vmin:vmax].
// ----------------------------------------------------------------------------
template <>
void UnitTest::Rand(
    vector<uint8_t> &v,
    const uint8_t &vmin,
    const uint8_t &vmax,
    const int& seed)
{
    UnitTest::Raw raw(seed);

    float factor = (float)(vmax - vmin) / UnitTest::Raw::VMAX;

    for (size_t i = 0; i < v.size(); i++)
        v[i] = vmin + (uint8_t)(raw.Next<uint8_t>() * factor);
}

// ----------------------------------------------------------------------------
// Initialize a size_t vector.
// Output range: [vmin:vmax].
// ----------------------------------------------------------------------------
template <>
void UnitTest::Rand(
    vector<size_t> &v,
    const size_t &vmin,
    const size_t &vmax,
    const int& seed)
{
    UnitTest::Raw raw(seed);

    float factor = (float)(vmax - vmin) / UnitTest::Raw::VMAX;

    for (size_t i = 0; i < v.size(); i++)
        v[i] = vmin + (size_t)(raw.Next<size_t>() * factor);
}
