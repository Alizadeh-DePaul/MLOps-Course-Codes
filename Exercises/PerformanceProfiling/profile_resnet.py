"""Section 2 driver — torch.profiler on a ResNet-18 forward pass.

Mirrors the PyTorch tutorial example. Run this once to confirm your
torch.profiler install works, then experiment with the TODOs to answer
the section's questions.

Run:
    python profile_resnet.py
"""
import torch
import torchvision.models as models
from torch.profiler import (
    ProfilerActivity,
    profile,
    tensorboard_trace_handler,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)

# Activities: profile CPU work always; add CUDA when a GPU is present so
# the same script runs on every student machine without code changes.
activities = [ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities.append(ProfilerActivity.CUDA)
    model = model.cuda()
    inputs = inputs.cuda()


# ---------------------------------------------------------------------------
# 2.1  Single forward pass with shape recording
# ---------------------------------------------------------------------------
print("=" * 70)
print("2.1  Single forward pass — sort by cpu_time_total, top 10")
print("=" * 70)
with profile(activities=activities, record_shapes=True) as prof:
    model(inputs)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


# ---------------------------------------------------------------------------
# 2.2  Group by input shape — does shape correlate with cost?
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("2.2  Group by input shape, top 30")
print("=" * 70)
print(
    prof.key_averages(group_by_input_shape=True).table(
        sort_by="cpu_time_total", row_limit=30
    )
)


# ---------------------------------------------------------------------------
# 2.3  Memory profiling
# ---------------------------------------------------------------------------
# TODO: (Section 2, sub-step 1)
# Re-run profile() with profile_memory=True and sort the table by
# self_cpu_memory_usage. Which operations allocate the most memory?
print("\n" + "=" * 70)
print("2.3  Memory profile — sort by self_cpu_memory_usage, top 10")
print("=" * 70)
with profile(activities=activities, profile_memory=True, record_shapes=True) as prof_mem:
    model(inputs)

print(prof_mem.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))


# ---------------------------------------------------------------------------
# 2.4  Export a Chrome / Perfetto trace
# ---------------------------------------------------------------------------
# After running, drag trace.json into ONE of:
#   - https://ui.perfetto.dev/        (recommended, especially for big files)
#   - chrome://tracing                (still works in Chromium browsers)
prof_mem.export_chrome_trace("trace.json")
print("\nWrote trace.json. Drop it into https://ui.perfetto.dev/ to inspect.")


# ---------------------------------------------------------------------------
# 2.5  Multi-iteration profile with tensorboard_trace_handler
# ---------------------------------------------------------------------------
# Single forward passes are noisy. Profile over several iterations and call
# prof.step() each time so the profiler can apply its schedule correctly.
# The same .pt.trace.json file produced here is consumable by BOTH
# TensorBoard (via the torch_tb_profiler plugin) AND Perfetto UI.
print("\n" + "=" * 70)
print("2.5  Multi-iteration profile -> log/resnet18/")
print("=" * 70)
with profile(
    activities=activities,
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=tensorboard_trace_handler("./log/resnet18"),
    record_shapes=True,
) as prof_iter:
    for _ in range(10):
        model(inputs)
        prof_iter.step()

print("Wrote .pt.trace.json files to ./log/resnet18/")
print("Visualize with EITHER of:")
print("  TensorBoard:  tensorboard --logdir=./log")
print("                then open http://localhost:6006/#pytorch_profiler")
print("  Perfetto:     open https://ui.perfetto.dev/ and drag in any")
print("                .pt.trace.json file from log/resnet18/")
