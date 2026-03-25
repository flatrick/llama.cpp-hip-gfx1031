## ADDED Requirements

### Requirement: Build script clones source and builds the ROCm image
`build.docker-rocm.sh` SHALL clone the llama.cpp source into `./llama.cpp-src/` (if not already present) and build the Docker/Podman image tagged `llama-cpp-gfx1031:latest` from `Dockerfile.rocm`.

#### Scenario: First-time build on clean checkout
- **WHEN** `./build.docker-rocm.sh` is run and `llama.cpp-src/.git` does not exist and the image `llama-cpp-gfx1031:latest` does not exist
- **THEN** the script clones the llama.cpp repo into `./llama.cpp-src/`, runs `docker build`, and exits 0 on success

#### Scenario: Skip clone when source already present
- **WHEN** `./build.docker-rocm.sh` is run and `llama.cpp-src/.git` already exists
- **THEN** the clone step is skipped and the build proceeds directly

#### Scenario: Skip build when image already present
- **WHEN** `./build.docker-rocm.sh` is run and `llama-cpp-gfx1031:latest` already exists and `--force` is not passed
- **THEN** the script prints a message indicating the image already exists and exits 0 without rebuilding

#### Scenario: Force rebuild with --force flag
- **WHEN** `./build.docker-rocm.sh --force` is run and the image already exists
- **THEN** the script runs `docker build` unconditionally, replacing the existing image

#### Scenario: Build failure exits non-zero
- **WHEN** `docker build` fails (e.g. network error, Dockerfile error)
- **THEN** the script exits with a non-zero exit code and prints the error to stderr

### Requirement: Run scripts fail fast when image is absent
`llama.server.Q4_K_M.docker-rocm.sh` and `llama.server.UD-Q4_K_XL.docker-rocm.sh` SHALL check for the `llama-cpp-gfx1031:latest` image before attempting `docker run`, and SHALL exit with a clear error if it is not found.

#### Scenario: Run script invoked without image present
- **WHEN** a `*.docker-rocm.sh` launcher is run and `llama-cpp-gfx1031:latest` does not exist
- **THEN** the script prints `ERROR: image llama-cpp-gfx1031:latest not found — run ./build.docker-rocm.sh first` (or equivalent) and exits 1

#### Scenario: Run script invoked with image present
- **WHEN** a `*.docker-rocm.sh` launcher is run and the image exists
- **THEN** `docker run` is invoked immediately with no build or clone steps

### Requirement: Build script contains no model-specific run logic
`build.docker-rocm.sh` SHALL contain only clone and image-build logic. It SHALL NOT invoke `docker run` or reference any model HuggingFace paths.

#### Scenario: Inspect build script for run invocations
- **WHEN** `grep -n 'docker run\|podman run' build.docker-rocm.sh` is executed
- **THEN** the command returns no matches (exit code 1)
