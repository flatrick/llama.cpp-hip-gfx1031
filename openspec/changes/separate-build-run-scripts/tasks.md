## 1. Create build script

- [x] 1.1 Create `build.docker-rocm.sh` with clone logic: if `llama.cpp-src/.git` absent, `git clone` the repo
- [x] 1.2 Add image-exists check: skip `docker build` unless `--force` / `-f` flag is passed
- [x] 1.3 Add `docker build -f Dockerfile.rocm -t llama-cpp-gfx1031:latest` invocation with `set -e` guard
- [x] 1.4 Mark script executable (`chmod +x build.docker-rocm.sh`)
- [x] 1.5 Verify: run `grep -n 'docker run\|podman run' build.docker-rocm.sh` returns no matches

## 2. Strip build logic from docker-rocm run scripts

- [x] 2.1 Remove clone block and `docker build` block from `llama.server.Q4_K_M.docker-rocm.sh`
- [x] 2.2 Add image-absent check to `llama.server.Q4_K_M.docker-rocm.sh`: exit 1 with message directing user to `build.docker-rocm.sh`
- [x] 2.3 Remove clone block and `docker build` block from `llama.server.UD-Q4_K_XL.docker-rocm.sh`
- [x] 2.4 Add image-absent check to `llama.server.UD-Q4_K_XL.docker-rocm.sh`: exit 1 with same message

## 3. Verify

- [ ] 3.1 Run `./build.docker-rocm.sh` on a system without the image; confirm image `llama-cpp-gfx1031:latest` is created
- [ ] 3.2 Remove the image (`docker rmi llama-cpp-gfx1031:latest`), run a `*.docker-rocm.sh` launcher, confirm it exits 1 with the expected error message
- [ ] 3.3 Re-run `./build.docker-rocm.sh` with image present (no `--force`); confirm it skips the build and exits 0
- [ ] 3.4 Run `./build.docker-rocm.sh --force`; confirm `docker build` runs again
