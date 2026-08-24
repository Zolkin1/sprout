// Copyright 2026 Zachary Olkin. All rights reserved.
#pragma once

// The parity fixtures, under the names the tests have always used.
//
// The reader itself moved into the library (g1_deploy/blob_pack.h) when a SECOND consumer
// needed it -- the clip pack the deploy runtime plays motion clips from, which is the same
// manifest+blobs format and must not grow a second reader. Nothing else changed; this keeps
// `g1_deploy::test::Fixtures` meaning what it did so the parity tests read the same.

#include <string>

#include "g1_deploy/blob_pack.h"

namespace g1_deploy::test {

using Blob = g1_deploy::Blob;
using Case = g1_deploy::Case;
using Fixtures = g1_deploy::BlobPack;

/// Where the committed fixtures live; see g1_deploy::DefaultFixtureDir.
inline std::string DefaultFixtureDir() { return g1_deploy::DefaultFixtureDir(); }

}  // namespace g1_deploy::test
