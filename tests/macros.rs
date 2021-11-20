use slimbox::*;

#[test]
fn fn_trait() {
    let specified = slimbox_unsize!(dyn Fn() -> i32, || 420);
    let inferred: SlimBox<dyn Fn() -> i32> = slimbox_unsize!(|| 42);

    let v = vec![specified, inferred];
    assert_eq!(v[0](), 420);
    assert_eq!(v[1](), 42);
}

#[test]
fn sized() {
    let specified = slimbox_unsize!(i32, 420);
    let inferred: SlimBox<i32> = slimbox_unsize!(42);

    let v = vec![specified, inferred];
    assert_eq!(*v[0], 420);
    assert_eq!(*v[1], 42);
}
