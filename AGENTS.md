# AGENTS.md

## Project Mission

nano-nccl is a deliberately narrow GPU collective communication library and an
NCCL differential performance research project. It focuses on one algorithm
and one protocol so that their implementation can be understood end to end:

- algorithm: Ring only
- protocol: Simple only
- collectives: out-of-place AllReduce, ReduceScatter, and AllGather
- transports: per-edge SHM, P2P, and NET
- NET backends: Socket reference path, host-pinned RDMA, and eventually
  GPUDirect RDMA
- device code: generated specialization registry backed by readable,
  handwritten CUDA templates

The primary engineering question is not merely whether nano-nccl is fast. The
project must explain why NCCL is fast on the scoped Ring + Simple path, where
nano-nccl is equivalent, where it is intentionally different, and which
differences cause measured performance gaps.

This is intended to be a solid, explainable systems project. Prefer depth,
causal evidence, and complete vertical paths over feature count.

## Scope Contract

The target supported matrix is:

- **Collectives**: AllReduce, ReduceScatter, AllGather
- **Buffer semantics**: out-of-place only; arbitrary overlap and in-place are
  unsupported
- **Dtypes**: `float`, FP16, BF16
- **Reduce ops**: `sum`, `avg`, `max`, `min`; AllGather has no reduce op
- **Algorithms**: Ring only
- **Protocols**: Simple only
- **Rank specializations**: 2, 4, and 8
- **API**: a small native C++ API; no NCCL API or ABI compatibility promise

Do not add Tree, CollNet, NVLS, LL, LL128, `double`, `int8`, `prod`, in-place
semantics, broadcast, or ranks outside 2/4/8 unless this contract is explicitly
revisited. Do not add placeholder abstractions for out-of-scope features.

nano-nccl is not a general NCCL replacement and must never be described as a
drop-in replacement.

## Current Implementation Snapshot

Keep the distinction between current behavior and target architecture clear:

- AllReduce is the only implemented collective. Public ReduceScatter and
  AllGather methods currently report unsupported operations.
- dtype and reduce op are compile-time kernel dimensions, but `nranks` is still
  a runtime kernel argument and `NANO_NCCL_NRANKS` fixes host-side sizing.
- Single-host SHM and P2P paths exist. Transport selection is resolved per Ring
  edge and can produce a mixed plan.
- MPI/socket and MPI/RDMA paths exist for cross-process AllReduce.
- RDMA currently uses registered host-pinned FIFO memory. SEND/RECV and
  WRITE+CTS modes exist; GPUDirect RDMA does not.
- `src/collective/collective.h` and `src/transport/transport.h` are empty virtual
  seams, not the target abstraction.
- The active Simple implementation lives under `src/transport/simple/`; a
  future ownership cleanup must be evidence-driven rather than a mechanical
  rename.

Do not claim target features as implemented until their correctness and
acceptance matrices pass.

## Research-First Development Order

Do not begin broad collective or architecture expansion before the scoped NCCL
mechanism model is established.

### Phase 1: Explain Existing AllReduce

Study NCCL and nano-nccl from public call through device and transport progress:

1. collective semantics and enqueue path
2. communicator, topology, channel planning, and kernel dispatch
3. generated function tables and compile-time specialization
4. Ring phase, chunk, channel, and slice work partition
5. Simple FIFO, step, credit, and persistent progress
6. device thread/warp roles, barriers, copy/reduce vectorization
7. memory ordering and GPU/device/host visibility
8. SHM, P2P, NET memory placement and progress
9. async failure, abort, teardown, and resource ownership

Phase 1 is complete only when:

- the NCCL AllReduce + Ring + Simple host-to-device path has a source map;
- every performance-critical nano-nccl mechanism is classified as equivalent,
  different, missing, extra, or unknown;
- confirmed causal claims have controlled A/B or profiling evidence;
- unresolved performance-critical unknowns have explicit experiment plans;
- current contract-size performance gaps can be explained without relying on a
  single benchmark run; and
- adopted and rejected NCCL mechanisms both have written rationale.

### Phase 2: Build the Scoped Library Architecture

Use Phase 1 evidence to implement typed collective descriptors, real Ring and
Simple boundaries, generated rank specializations, ReduceScatter, AllGather,
and a shared composition for AllReduce.

### Phase 3: Deepen NET

Retain Socket as a reference NET data-plane backend, characterize host-pinned
RDMA, then implement and compare GPUDirect RDMA. Compare host-pinned paths with
NCCL GDR disabled and GDR paths with NCCL GDR enabled.

## Local Workspace And Worktrees

Machine-specific paths are stored in an untracked workspace registry. Locate it
with:

```bash
git config --local --get nano-nccl.workspaceConfig
```

The configured file is normally named `.local/workspace.yaml` in the primary
nano-nccl checkout. It records the local nano-nccl, NCCL, nccl-tests, and
experiments repositories, the external worktree root, and the raw artifact
root. Never copy its absolute paths into tracked files.

All development, rebuilds, commit switching, and temporary experiments for
nano-nccl, NCCL, and nccl-tests must run in dedicated Git worktrees:

```text
<worktree-root>/<repository>/<branch-name>/
```

Rules:

- Primary checkouts are coordination points only. Do not switch their commits,
  build in them, or apply experimental patches there.
- Create a dedicated worktree before modifying source, rebuilding a reference
  project, or running a commit-specific experiment.
- Inspect `git status` and `git worktree list` before acting.
- Treat unknown changes as user work. Never discard, reset, clean, or overwrite
  them.
- Switching detached commits and applying temporary instrumentation are allowed
  only in the worktree created for that experiment.
- Do not delete worktrees or branches unless explicitly requested.
- Build directories and large captures stay in the worktree or configured raw
  artifact root, never in a primary checkout.
- If the local registry is unavailable, ask for the local setup instead of
  guessing absolute paths.

The NCCL and nccl-tests source trees are research inputs, not nano-nccl build
dependencies.

## Evidence And NCCL Differential Analysis

Experiment records live in the configured experiments repository under its
`nano-nccl/` domain and follow that directory's `AGENTS.md`. Large raw logs,
Nsight captures, binaries, and generated files live under the configured local
artifact root.

Every performance-oriented change must have this evidence chain:

```text
GAP-ID
  -> hypothesis
  -> NCCL source reference
  -> nano-nccl source reference
  -> controlled experiment
  -> correctness result
  -> profiling or counter evidence
  -> repeated performance result
  -> keep, reject, or remain-unknown decision
```

Source references include repository, commit SHA, relative file path, and
symbol. Never claim how NCCL works from memory when the configured NCCL source
can be inspected.

Use these gap classifications:

- `Equivalent`: mechanism and relevant semantics are equivalent.
- `Different`: same purpose, materially different implementation.
- `Missing`: NCCL has a scoped mechanism that nano-nccl lacks.
- `Extra`: nano-nccl has a mechanism not present in the scoped NCCL path.
- `Unknown`: evidence is insufficient.

Also label each difference as correctness-critical, performance-critical,
engineering trade-off, or intentional scope difference.

Implementation difference alone does not establish performance causality. NCCL
having a mechanism does not imply nano-nccl should copy it. Failed experiments
are durable results and must be recorded so they are not repeated.

## Target Architecture

The target data flow is:

```text
typed collective descriptor
  -> semantic validation and normalization
  -> collective lowering
  -> Ring schedule and work partition
  -> generated specialization registry
  -> handwritten Ring + Simple kernel template
  -> per-edge SimpleConnectionView
  -> SHM | P2P | NET(Socket | host-pinned RDMA | GDR)
```

### Collective Semantics

The semantic layer describes what the operation means, not how to execute it.
Use distinct typed descriptors for AllReduce, ReduceScatter, and AllGather.
They own buffer layout, element counts, dtype, applicable redop, streams, and
out-of-place validation.

Do not use a generic bag of optional fields and do not encode Ring phases,
channels, Simple steps, transport policy, or kernel specialization in public
descriptors. If internal unified dispatch is needed, use a typed variant of
normalized tasks.

### Ring And Simple

- Collective lowering maps a normalized task to Ring phases.
- Ring owns rank order, phase ordering, channel/chunk work partition, and
  collective-specific chunk ownership.
- Simple owns FIFO layout, slice/step geometry, credit, persistent directional
  progress, memory ordering, and device send/recv/reduce primitives.
- Collective code must not duplicate Simple primitives.
- Simple must not own transport allocation or NET progress.

### Transport

Transport answers how one directed Ring edge stores, exposes, and progresses
Simple data. It does not know collective semantics.

- Resolve transport per directed edge; mixed SHM/P2P/NET Rings are supported.
- SHM owns mapped host storage and mapping.
- P2P owns peer device storage, peer mapping, and capability checks.
- NET is a first-class transport family. Socket, host-pinned RDMA, and GDR are
  NET backends or data-plane modes.
- Backends own connection establishment, memory placement/registration,
  proxy/progress, capability checks, and teardown.
- Kernels consume one transport-neutral device-side `SimpleConnectionView`.
- Transport backend must not become a kernel template dimension.
- `auto` may select according to topology and capabilities. An explicitly
  requested backend must fail rather than silently fall back.
- Report the actual backend of every directed edge; an aggregate `mixed` label
  alone is insufficient.

### Kernel Generation

Handwrite readable CUDA templates. Generate only explicit instantiations,
launch wrappers, and the runtime registry.

The valid specialization dimensions are:

```text
collective x dtype x applicable redop x nranks
```

The default rank set is exactly 2, 4, and 8. AllGather has no redop dimension.
Transport and channel count are not specialization dimensions. Rank is a
non-type template parameter so Ring phases and rank arithmetic can be compiled
and unrolled for each supported rank.

Generated files belong in the build directory and are not committed. A missing
specialization must fail during communicator setup or dispatch and list the
compiled rank set. Build reporting should expose generated combination count,
compile time, and binary size so specialization does not grow without review.

## Error Model

- Invalid descriptors, unsupported rank specialization, and unavailable
  explicitly selected backends fail synchronously before launch.
- Asynchronous NET/proxy failures latch the communicator's first stable error
  and publish a GPU-visible abort so kernels do not spin forever.
- After a latched async error, the communicator rejects new collective work.
- `check_async_error()` reports the stable root error.
- Teardown is idempotent and handles partial initialization and asynchronous
  failure.
- Mixed-plan errors identify the directed edge and actual backend.
- All CUDA API calls use checked error handling.

## Testing Strategy

Required coverage includes:

- typed descriptor count, layout, dtype/redop, stream, and out-of-place rules;
- Ring phase and chunk ownership for ranks 2/4/8 and boundary message sizes;
- Simple FIFO layout, step/credit persistence, ordering, and abort behavior;
- generator completeness for valid combinations and rejection of invalid ones;
- SHM, P2P, Socket, host-pinned RDMA, GDR, and mixed edge plans;
- correctness for every combination claimed as supported;
- differential outputs against NCCL for matching semantics;
- bootstrap, peer disconnect, proxy failure, partial initialization, and cleanup
  fault injection; and
- performance regression and NCCL-relative acceptance campaigns.

Only claim combinations tested on the stated hardware, rank count, and
transport. Generated code without execution coverage is not validated support.

## Performance Acceptance

Use `-w 5 -n 20` for acceptance measurements. Candidate binaries must have
profiling and investigation-only instrumentation disabled. Observability builds
cannot provide acceptance numbers.

NCCL comparisons must be same-round and match all relevant conditions:

- Ring algorithm and Simple protocol
- collective and out-of-place semantics
- ranks, dtype, redop, and message sizes
- channels and buffer size
- topology and transport class
- GDR disabled for host-pinned comparison, enabled for GDR comparison

For each declared contract matrix:

- nano-nccl/NCCL busbw geomean must be at least `0.95`;
- every contract message-size ratio must be at least `0.90`; and
- regression against the accepted nano-nccl baseline must not exceed `3%`.

Every result below NCCL requires a written causal explanation or remains an
open `Unknown` gap. Per-size parity or wins are stretch results, not the sole
definition of project correctness. Correctness must pass before performance is
interpreted.

Historical performance tables predate this contract unless explicitly
revalidated under it.

## Sensitive Information

Do not write hostnames, IP addresses, physical interface names, MAC addresses,
GPU UUIDs, absolute user paths, credentials, or tokens to files that can be
committed. Use placeholders such as `<host-a>`, `<interface>`,
`<path-to-nccl-lib>`, and `<local-artifact-root>`.

The untracked workspace registry may contain local absolute paths. Raw local
artifacts may contain machine details, but they must remain outside tracked
content. Before committing experiment-derived material, scan it for sensitive
values.

## Coding Standards

- Indent with 4 spaces; do not use tabs.
- Use `#pragma once` for headers.
- Include order: own header, C++ standard library, CUDA, then system headers.
- Use `CUDA_CHECK_THROW` and `throw std::runtime_error`; check every CUDA API
  call.
- Classes and structs use PascalCase; functions and files use snake_case;
  constants use kPascalCase; namespaces mirror directory ownership.
- Comments explain non-obvious reasons, invariants, ordering, or ownership.
  Do not comment obvious mechanics.
- Prefer the smallest change that can test the current hypothesis.
- Do not add compatibility layers, generic factories, or virtual interfaces
  without a concrete current consumer.

## Build Snapshot

The current implementation still uses one build-time rank count:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DNANO_NCCL_NRANKS=<rank-count> \
  -DNANO_NCCL_CUDA_ARCH=<cuda-arch>
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

MPI/RDMA currently requires:

```bash
cmake -S . -B build-rdma -DCMAKE_BUILD_TYPE=Release \
  -DNANO_NCCL_ENABLE_MPI=ON \
  -DNANO_NCCL_ENABLE_RDMA=ON \
  -DNANO_NCCL_NRANKS=<global-rank-count> \
  -DNANO_NCCL_CUDA_ARCH=<cuda-arch>
cmake --build build-rdma -j$(nproc)
```

The specialization generator will replace the single-rank build contract only
after its design is justified by Phase 1 evidence and implemented with registry
tests.

`float` and FP16 require SM70+ because the current Simple counters use
system-scope ordering. BF16 requires SM80+. `NANO_NCCL_CUDA_ARCH` defaults to 70;
CMake rejects values below 70. Distributed builds require matching Open MPI
4.1.2 C ABI and the same nano-nccl commit on every host.

### Build / ownership contracts

Current module ownership on the Ring + Simple path (docs contract for static
checks; not a substitute for Phase 1 evidence):

- `src/transport/simple/` owns FIFO layout, step ordering, and slice geometry.
  Persistent progress uses distinct send and recv base steps per rank/channel.
- `src/transport/shm/` owns mapped host FIFO storage/control only.
- `src/collective/all_reduce/ring_simple_geometry.h` owns Ring channel/rank work partitioning.
- Transport runtime lifecycle and orchestration remain in `Communicator::Impl`.

## Communication And Resume Claims

Project documentation and resume statements must separate:

- implemented and validated behavior;
- target architecture;
- measured NCCL-equivalent mechanisms;
- intentional differences;
- confirmed performance gaps; and
- unresolved hypotheses.

Prefer claims such as "identified and closed a system-fence publication gap
with source-guided A/B evidence" over broad claims such as "reimplemented
NCCL". A result is credible only when another engineer can trace it from claim
to source references, experiment, correctness, measurements, and decision.
