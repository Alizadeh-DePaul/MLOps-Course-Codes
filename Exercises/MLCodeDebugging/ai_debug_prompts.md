# AI-assisted debugging prompt cheat sheet

A short, curated list of Copilot Chat prompts that work well for the kind of
ML bugs in `vae_mnist_buggy.py`. The point isn't to memorize them — it's to
notice the *shape* of a useful prompt versus a lazy one.

The shape that works:

> 1. **Context** — paste the relevant code window, not the whole file.
> 2. **Symptom** — paste the exception or describe the unexpected behavior literally.
> 3. **Hypothesis or constraint** — what you've already ruled out, what you expect to see.
> 4. **Ask** — "find the bug" or "explain why X happens", not "fix this for me".

The shape that doesn't:

> "fix my code" — you'll get a confident answer that may or may not be the real bug.

## Slash commands worth knowing

- `/explain` — paste a confusing block; ask for a step-by-step walkthrough.
- `/fix` — select code, type `/fix`. Best when paired with the exception text.
- `/help` — list every slash command Copilot Chat supports in your version.

## Prompt 1 — when a stack trace makes no sense

When you hit a `RuntimeError: mat1 and mat2 shapes cannot be multiplied (100x400 and 20x784)`
and aren't sure which `nn.Linear` is at fault:

```
[Paste the full traceback]

[Paste only the Decoder class definition]

The traceback says shapes (100x400) and (20x784) can't be multiplied.
The decoder runs `FC_hidden` then `FC_output` on the result of FC_hidden.
Which Linear layer's `in_features` argument is inconsistent with the
shape of the tensor flowing into it? Don't fix anything yet, just
explain which line is wrong and why.
```

Why this works: you're constraining Copilot to *diagnose* before it *patches*.
A naive `/fix` here often returns "use `latent_dim` everywhere" or "reshape
the tensor", both of which paper over the real shape mismatch.

## Prompt 2 — when training "runs" but loss never decreases

```
[Paste the training loop only]

The script runs without errors. Loss starts at ~25000 per batch and stays
at ~25000 per batch for all 20 epochs. Reconstructions look like noise.

I've already verified:
  - The data loader returns the right shape and dtype.
  - The model has trainable parameters (model.parameters() is non-empty).
  - The loss is differentiable (it's a sum of BCE + KLD on float tensors).

Walk me through what the *training loop* must do every iteration for
gradient descent to actually descend. Compare your list against the
loop above and tell me what's missing.
```

Why this works: the prompt forces Copilot to enumerate "the canonical
training loop" before pattern-matching against your specific code, which
makes it reliably catch the missing `optimizer.zero_grad()` rather than
suggesting a learning-rate change.

## Prompt 3 — when a math line "looks fine" but the model never converges

```
[Paste Encoder.reparameterization]

This is supposed to implement the VAE reparameterization trick:
  z = mean + std * epsilon

The function takes (mean, var) but the call site passes (mean, log_var).
What is the correct relationship between log_var and the std you should
multiply epsilon by? Is the function above implementing it correctly?
Answer in two sentences.
```

Why this works: you're being explicit about what the *correct* math is, so
Copilot can compare instead of guess. Without that anchor, Copilot will
sometimes claim the buggy version is "fine because var is just a scaling
factor" — that's the model rationalizing rather than checking.

## Prompt 4 — when a CUDA tensor and a CPU tensor collide

```
[Paste Encoder.reparameterization]

I'm getting:
  RuntimeError: Expected all tensors to be on the same device, but found
  at least two devices, cuda:0 and cpu!

Inside this function, `mean` is on cuda:0 (it came from a Linear layer on
the model). `epsilon` is created here. What's the device of a tensor
created with torch.randn(...) by default? How do I make it inherit the
device of an existing tensor instead?
```

Why this works: again, anchoring. You're not asking "fix the device
error", you're asking the question whose answer *is* the fix
(`torch.randn_like(...)` instead of `torch.randn(...)`).

## When NOT to ask Copilot

- **You haven't read the traceback yet.** Read it first. Half the time the answer is on the second-to-last line.
- **You can answer the question by adding one `print(t.shape, t.device)` line.** A debugger or a print is faster than a chat round-trip.
- **You don't have a hypothesis.** Asking "what's wrong with my code?" with 200 lines pasted gets you a stylistic critique, not a bug fix. Build the hypothesis with a debugger first; *then* ask Copilot to confirm or refute it.

## Verification step

Whatever Copilot tells you, set a breakpoint at the line it says is wrong
and *look at the actual values* before you accept the fix. AI debuggers
hallucinate. Real ones don't.
