##
This is a one-to-one numpy reimplementation of the rust script.

Notes:

- I had a working translation of the rust version in 30 min
- started without mypy because typing np shapes isn't super easy
- encountered multiple errors that could have been caught by a type checker:
  - positional function arguments were swapped, caught at first test run
  - index err at runtime, state mutated accidentally to the wrong shape (must be mutable though). I took argmin along the wrong axis, producing wrong array shape- no warning)
  - I didn't specify an axis when averaging clusters, got a scalar instead of vector
- A good IDE is essential. I stated off in VSCode because I want to like it, but I get so little assistance from it I immediately switched back to PyCharm. 

It took me a total or 5 runs of the unit test, 10-15 min to fix all the above. I'm on the fence whether python or rust version would be easier to maintain. I find the python one easier to read because it's shorter and because I'm much more experienced with numpy slicing- but maybe there's too much magic and others may find this harder. With a small investment into typing I could probably get something that's self-documenting like rust


## Rust
`cargo install flamegraph && sudo cargo flamegraph`

`rm -rf python/rsmeans/rsmeans.cpython-310-darwin.so` to clear stale rust wheels


Speed comparison for 10k x 1.5k matrix w/ 10 clusters (M1 pro macbook pro):
- numpy 1.2s
- rust 0.25s from `cargo test -r -- --nocapture` (27s when running a debug build without `-r`)
- c 1.3s
- sklearn 0.4s

Intel macbook pro:
- numpy 4s
- rust 0.7s/ parallel rust 0.2s
- c 1.9s
- sklearn 0.4s

Flamegraph shows most of the time is spent in euclidean distance calculation (map a->a*a, subtract, sum)
