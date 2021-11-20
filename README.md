# `slimbox`

A thin pointer type for heap allocation that stores metadata together with the value, in the same allocation.

`slimbox` is `no_std`-compatible (requiring `alloc`).

The MSRV (Minimum Supported Rust Version) is 1.56, for edition 2021. You can enable the `nightly` or `unsafe_stable` features for extra functionality; the `nightly` feature requires a nightly toolchain (as of 2021-11-17), the `unsafe_unstable` feature assumes that (potentially-)wide pointers contain a thin pointer at offset 0 - this has been the case for a long time and it shouldn't really change in the future, but it's still an assumption that's not backed by stability guarantees.

## License

Licensed under either of
 * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
