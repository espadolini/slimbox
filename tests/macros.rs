use slimbox::*;

#[test]
fn fn_trait() {
    let b = slimbox_unsize!(dyn Fn() -> i32, || 42);

    assert_eq!(b(), 42);
}

#[test]
fn sized() {
    let b = slimbox_unsize!(i32, 8);
    assert_eq!(*b, 8);
}
