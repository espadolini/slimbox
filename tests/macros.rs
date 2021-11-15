use slimbox::*;

#[test]
fn fn_trait() {
    let macro_declared_inferred: SlimBox<dyn Fn() -> i32> = slimbox_unsize!(|| 42);
    let macro_declared_typed = slimbox_unsize!(dyn Fn() -> i32, || 42);

    assert_eq!(macro_declared_inferred(), 42);
    assert_eq!(macro_declared_typed(), 42);
}

#[test]
fn sized() {
    let b: SlimBox<i32> = slimbox_unsize!(8);
    assert_eq!(*b, 8);
}
