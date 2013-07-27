#pragma once
#include <pcl/PolygonMesh.h>
#include <vector>
#include "macros.h"

CLOUDPROC_API std::vector<pcl::PolygonMesh::Ptr> ConvexDecompHACD(const pcl::PolygonMesh& mesh, float concavity);
