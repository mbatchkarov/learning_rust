# Learning Rust

This is a toy implementation of k-means clustering in Rust using `nalgebra`. The point is to learn some Rust and to get a feel for the ergonomics of the language by implementing the same algorithm in [C](https://github.com/mbatchkarov/relearning_c) and Python (with Numpy, coming soon). Please don't use this code.

# Impressions

- guardrails everywhere, clear compiler error messages
- nice functional patterns (map, fold)
- macros are cool `!vec` and `ndarray`'s `!s`, can enhance the syntax of the language- but this can go to far (link below)!
- much steeper learning curve than C. C has no hidden magic so the code ends up being a little longer (and there less legible, IMHO) but it's clear at all times. For the C implementation I mostly looked at the docs of `GSL` (which are very nice). For Rust I spent a log of time on google to get the basics to work. Just check [this](https://docs.rs/ndarray/latest/ndarray/macro.s.html) out- ugh.
- the wrong problem for Rust- the script is too simple to benefit from the compiler's guardrails (eg I don't care about memory leaks) but I have to address them. Does not save you from logical errors like accidentally transposing a matrix. Would be much more useful in a larger codebase where quality is needed.
- Numpy's ergonomics are hard to beat- rich functionality (eg ndarray does not have argmin as far and I can tell), concise and obvious slicing, easy IO, etc.