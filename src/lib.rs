//! [`SlimBox`] is a thin pointer type for heap allocation; by storing metadata
//! (either a full pointer, or just the metadata when the `nightly` feature is
//! enabled) together with the value in the same allocation, it allows you to
//! box a [DST] (i.e. a slice or a trait object) while only using one machine
//! word's worth of space. Additionally, the [`SlimRef`] and [`SlimMut`] types
//! provide the equivalent of shared and exclusive references to the contents of
//! a `SlimBox`, while also being the size of a thin pointer.
//!
//! A convenient [`slimbox_unsize!`] macro is provided, to construct a
//! `SlimBox<T>` from a value of type `S` that can be [unsized] to `T`. By
//! enabling the `nightly` feature (requiring a nightly compiler) or the
//! `unsafe_stable` feature, the [`SlimBox::from_box`] conversion function is
//! also made available, to convert from a preexisting `Box<T>` into a
//! `SlimBox<T>`.
//!
//! The use of the `unsafe_stable` feature does rely on something that could
//! potentially become false in the future, leading to UB; namely, the fact that
//! a fat pointer contains a thin pointer at offset 0. This does seem to be the
//! case now, and it's probably going to stay like this in the future, but for
//! additional peace of mind it is recommended to just construct the `SlimBox`
//! out of the original `Sized` value in the first place.
//!
//! The three types are FFI-compatible and null-optimized, so you can declare
//! `extern "C"` functions that deal in `SlimBox`, `SlimRef` and `SlimMut` or
//! [`Option`]s of those while only declaring opaque pointers to the foreign
//! side.
//!
//! ```
//! # use slimbox::{slimbox_unsize, SlimBox, SlimRef};
//! trait Frob {
//!     fn foo(&self) -> i32;
//! }
//! # struct A(i32);
//! # struct B(i32);
//! # impl Frob for A { fn foo(&self) -> i32 { 0 } }
//! # impl Frob for B { fn foo(&self) -> i32 { 0 } }
//!
//! // void *frob_new(int32_t p);
//! #[no_mangle]
//! extern "C" fn frob_new(p: i32) -> Option<SlimBox<dyn Frob>> {
//!     if p == 0 {
//!         None
//!     } else if p % 2 == 0 {
//!         Some(slimbox_unsize!(A(p / 2)))
//!     } else {
//!         Some(slimbox_unsize!(B(p)))
//!     }
//! }
//!
//! // int32_t frob_foo(const void *f);
//! #[no_mangle]
//! extern "C" fn frob_foo(f: SlimRef<'_, dyn Frob>) -> i32 {
//!     f.foo()
//! }
//!
//! // void frob_free(void *f);
//! #[no_mangle]
//! extern "C" fn frob_free(f: Option<SlimBox<dyn Frob>>) {}
//! ```
//!
//! [DST]: https://doc.rust-lang.org/reference/dynamically-sized-types.html
//! [unsized]: core::marker::Unsize

#![warn(missing_docs)]
#![cfg_attr(feature = "nightly", feature(ptr_metadata, set_ptr_value))]
#![cfg_attr(doc, feature(doc_cfg))]
#![no_std]
extern crate alloc;

use alloc::boxed::Box;
use core::{
    ffi::c_void,
    marker::PhantomData,
    mem::{self, MaybeUninit},
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

/// [`pointer::set_ptr_value`][set_ptr_value] in a function, so it can be
/// replaced when nightly features are not available.
///
/// [set_ptr_value]: https://doc.rust-lang.org/std/primitive.pointer.html#method.set_ptr_value
#[cfg(feature = "nightly")]
fn set_ptr_value<T: ?Sized>(ptr: *mut T, val: *mut u8) -> *mut T {
    ptr.set_ptr_value(val)
}

/// Stable version of [`pointer::set_ptr_value`][set_ptr_value] that does enough
/// checks to hopefully make it loudly fail if and when the layout of `*mut T`
/// for `T: !Sized` changes.
///
/// This could technically trigger UB if the layout of trait objects (or unsized
/// pointers in general) changes such that the memory at offset 0 in a fat
/// pointer has validity requirements; however, miri would notice that pretty
/// quickly.
///
/// [set_ptr_value]: https://doc.rust-lang.org/std/primitive.pointer.html#method.set_ptr_value
#[cfg(all(feature = "unsafe_stable", not(feature = "nightly")))]
fn set_ptr_value<T: ?Sized>(mut ptr: *mut T, val: *mut u8) -> *mut T {
    assert!(mem::size_of::<*mut T>() >= mem::size_of::<*mut u8>());
    assert!(mem::align_of::<*mut T>() >= mem::align_of::<*mut u8>());

    let fake_ptr_a = 1 as *mut u8;
    let fake_ptr_b = 2 as *mut u8;

    let thin = &mut ptr as *mut *mut T as *mut *mut u8;

    unsafe {
        thin.write(fake_ptr_a);
        assert_eq!(ptr as *mut u8, fake_ptr_a);
        thin.write(fake_ptr_b);
        assert_eq!(ptr as *mut u8, fake_ptr_b);

        thin.write(val);
        assert_eq!(ptr as *mut u8, val);
    };

    ptr
}

/// [`alloc::alloc::alloc`] but returns a dangling aligned pointer on a
/// zero-sized allocation.
#[cfg(any(feature = "unsafe_stable", feature = "nightly"))]
fn alloc_extended(layout: core::alloc::Layout) -> *mut u8 {
    if layout.size() != 0 {
        let storage = unsafe { alloc::alloc::alloc(layout) };
        if storage.is_null() {
            alloc::alloc::handle_alloc_error(layout);
        }
        storage
    } else {
        layout.align() as *mut u8
    }
}

/// [`alloc::alloc::dealloc`] but does nothing on a zero-sized allocation.
#[cfg(any(feature = "unsafe_stable", feature = "nightly"))]
unsafe fn dealloc_extended(ptr: *mut u8, layout: core::alloc::Layout) {
    if layout.size() != 0 {
        alloc::alloc::dealloc(ptr, layout);
    }
}

/// impls Copy and Clone for a type without putting bounds on the generic type
/// parameter assumed to be `T: ?Sized`
macro_rules! impl_unbound_copy {
    ($T:ty) => {
        impl<T: ?Sized> Copy for $T {}
        impl<T: ?Sized> Clone for $T {
            fn clone(&self) -> Self {
                *self
            }
        }
    };
}

/// A container for a value that can also hold enough information to be able to
/// reconstruct a (potentially-wide) pointer to itself from a thin pointer to
/// itself. The internal container type used by [`SlimBox`].
#[repr(C)]
pub struct Slimmable<T: ?Sized, S: ?Sized = T> {
    /// Enough space for a `SlimAnchor` for `Slimmable<T, S>` as `Slimmable<T, T>`.
    anchor: MaybeUninit<SlimAnchor<T>>,
    /// The wrapped value.
    pub value: S,
}

impl<T: ?Sized, S> Slimmable<T, S> {
    /// Make a `Slimmable<T, S>` out of a `value` of type `S`.
    pub fn new(value: S) -> Slimmable<T, S> {
        Slimmable {
            anchor: MaybeUninit::uninit(),
            value,
        }
    }

    /// Make a `Box<Slimmable<T, S>>` out of a `value` of type `S`.
    pub fn boxed(value: S) -> Box<Slimmable<T, S>> {
        Box::new(Slimmable::new(value))
    }
}

/// The first field of a [`Slimmable`], containing enough information to be able
/// to reconstruct a `*mut Slimmable<T>` from a `*mut SlimAnchor<T>`.
struct SlimAnchor<T: ?Sized>(
    #[cfg(feature = "nightly")] <Slimmable<T> as core::ptr::Pointee>::Metadata,
    #[cfg(not(feature = "nightly"))] *mut Slimmable<T>,
);
impl_unbound_copy! { SlimAnchor<T> }

// SAFETY: our usage of the anchor will depend on wrapper types' bounds anyway
unsafe impl<T: ?Sized> Send for SlimAnchor<T> {}
unsafe impl<T: ?Sized> Sync for SlimAnchor<T> {}

impl<T: ?Sized> SlimAnchor<T> {
    /// Extracts a `SlimAnchor` from `ptr`
    fn new(ptr: *mut Slimmable<T>) -> SlimAnchor<T> {
        #[cfg(feature = "nightly")]
        {
            SlimAnchor(core::ptr::metadata(ptr))
        }

        #[cfg(not(feature = "nightly"))]
        {
            SlimAnchor(ptr)
        }
    }

    /// Rebuilds a `*mut Slimmable<T>` given `self` and a `ptr` (pointing to `self`).
    fn get_ptr(self, ptr: *mut SlimAnchor<T>) -> *mut Slimmable<T> {
        #[cfg(feature = "nightly")]
        {
            core::ptr::from_raw_parts_mut(ptr.cast(), self.0)
        }

        #[cfg(not(feature = "nightly"))]
        {
            let _ = ptr;
            self.0
        }
    }
}

/// A non-null thin pointer to a [`Slimmable`], by virtue of pointing to its
/// first field, a (potentially-uninit) [`SlimAnchor`].
#[derive(Debug)]
#[repr(transparent)]
struct SlimPtr<T: ?Sized>(NonNull<SlimAnchor<T>>);
impl_unbound_copy! { SlimPtr<T> }

// SAFETY: we make SlimPtr Send and Sync so that our Box/Ref/Mut types are
// automatically bound by their PhantomData
unsafe impl<T: ?Sized> Send for SlimPtr<T> {}
unsafe impl<T: ?Sized> Sync for SlimPtr<T> {}

impl<T: ?Sized> SlimPtr<T> {
    /// Creates a new `SlimPtr` if `ptr` is non-null.
    fn new(ptr: *mut Slimmable<T>) -> Option<SlimPtr<T>> {
        NonNull::new(ptr as *mut SlimAnchor<T>).map(SlimPtr)
    }

    /// Creates a new `SlimPtr`.
    ///
    /// # Safety
    ///
    /// `ptr` must be non-null.
    unsafe fn from_raw_unchecked(ptr: *mut c_void) -> SlimPtr<T> {
        SlimPtr(NonNull::new_unchecked(ptr as *mut SlimAnchor<T>))
    }

    /// Reconstructs a regular (potentially wide) pointer to the pointed
    /// [`Slimmable<T>`].
    ///
    /// # Safety
    ///
    /// `self` must point to a valid `Slimmable<T>`.
    unsafe fn as_ptr(self) -> *mut Slimmable<T> {
        let ptr = self.0.as_ptr();
        (*ptr).get_ptr(ptr)
    }

    /// Reconstructs a shared reference to the pointed [`Slimmable<T>`], with
    /// arbitrary lifetime.
    ///
    /// # Safety
    ///
    /// `self` must point to a valid `Slimmable<T>` and it must be legal to
    /// build a shared reference to it.
    unsafe fn as_ref<'a>(self) -> &'a Slimmable<T> {
        &*self.as_ptr()
    }

    /// Reconstructs an exclusive reference to the pointed [`Slimmable<T>`],
    /// with arbitrary lifetime.
    ///
    /// # Safety
    ///
    /// `self` must point to a valid `Slimmable<T>` and it must be legal to
    /// build an exclusive reference to it.
    unsafe fn as_mut<'a>(self) -> &'a mut Slimmable<T> {
        &mut *self.as_ptr()
    }
}

/// A thin owned pointer; like [`Box<T>`] but guaranteed to be as big as a
/// regular pointer, even for `!Sized` types.
///
/// Internally this points to a [`Slimmable<T>`].
#[derive(Debug)]
#[repr(transparent)]
pub struct SlimBox<T: ?Sized> {
    /// Owned thin pointer to a `Slimmable`.
    slim_box: SlimPtr<T>,
    /// Marker to act as if we have a `Box<T>`.
    _phantom: PhantomData<Box<T>>,
}

impl<T: ?Sized> SlimBox<T> {
    /// Repacks a `Box<Slimmable<T>>` into a `SlimBox<T>`, thinning its pointer.
    pub fn from_boxed_slimmable(boxed: Box<Slimmable<T>>) -> SlimBox<T> {
        let ptr = Box::into_raw(boxed);

        // SAFETY: ptr points to a Slimmable<T>
        unsafe { &mut (*ptr).anchor }.write(SlimAnchor::new(ptr));

        // the following cast is sound because Slimmable<T> is repr(C) so a
        // pointer to it can be cast to a pointer to its first field
        SlimBox {
            slim_box: SlimPtr::new(ptr).unwrap(),
            _phantom: PhantomData,
        }
    }

    /// Repacks a `SlimBox<T>` into a `Box<Slimmable<T>>`.
    pub fn into_boxed_slimmable(self) -> Box<Slimmable<T>> {
        // SAFETY: self.slim_box points to a valid Slimmable<T>
        let ptr = unsafe { self.slim_box.as_ptr() };
        mem::forget(self);
        // SAFETY: we were originally a Box<Slimmable<T>>
        unsafe { Box::from_raw(ptr) }
    }

    /// Moves `value` in a new `SlimBox<T>`. `T` must be `Sized` to use this.
    pub fn new(value: T) -> SlimBox<T>
    where
        T: Sized,
    {
        SlimBox::from_boxed_slimmable(Slimmable::boxed(value))
    }

    /// Moves the value contained in `boxed` into a `SlimBox`. This function
    /// will likely make a new allocation.
    ///
    /// # Panics
    ///
    /// Will panic if prepending the required metadata to the allocation would
    /// overflow the layout.
    #[cfg(any(feature = "nightly", feature = "unsafe_stable", doc))]
    #[cfg_attr(doc, doc(cfg(any(feature = "nightly", feature = "unsafe_stable"))))]
    pub fn from_box(boxed: Box<T>) -> SlimBox<T> {
        use core::alloc::Layout;

        // we manually build the Layout for a Slimmable<T> that can hold the
        // currently boxed value
        let slimmable_layout = Layout::new::<SlimAnchor<T>>();
        let value_layout = Layout::for_value(boxed.deref());
        let (slimmable_layout, value_offset) = slimmable_layout.extend(value_layout).unwrap();
        let slimmable_layout = slimmable_layout.pad_to_align();

        if slimmable_layout == value_layout && value_offset == 0 {
            let ptr = Box::into_raw(boxed) as *mut Slimmable<T>;
            // SAFETY: SlimAnchor<T> is a MaybeUninit ZST, it's sound to just
            // cast the Box<T> into a Box<Slimmable<T>>
            let boxed = unsafe { Box::from_raw(ptr) };
            return SlimBox::from_boxed_slimmable(boxed);
        }

        let slimmable_ptr = alloc_extended(slimmable_layout);
        let value_ptr = Box::into_raw(boxed);

        // SAFETY: we're initializing the newly-allocated Slimmable<T> that
        // lives at slimmable_ptr, moving the T from the allocation at value_ptr
        // (the anchor field is MaybeUninit so we don't have to touch it)
        unsafe {
            core::ptr::copy_nonoverlapping(
                value_ptr as *const u8,
                slimmable_ptr.add(value_offset),
                value_layout.size(),
            );
            dealloc_extended(value_ptr as *mut u8, value_layout);
        }

        // copy the T metadata from the pointer to T
        let slimmable_ptr = set_ptr_value(value_ptr, slimmable_ptr) as *mut Slimmable<T>;

        // SAFETY: slimmable_ptr points to a globally-allocated, owned,
        // initialized and valid Slimmable<T>, so we can pass it to a Box
        SlimBox::from_boxed_slimmable(unsafe { Box::from_raw(slimmable_ptr) })
    }

    /// Returns a `*mut c_void` pointing to the internal [`Slimmable<T>`], which
    /// can be conveniently passed over FFI boundaries that expect a pointer
    /// type, and used later with [`SlimRef::from_raw`] or
    /// [`SlimMut::from_raw`].
    pub fn as_raw(&self) -> *mut c_void {
        self.slim_box.0.as_ptr() as *mut c_void
    }

    /// Consumes the box and returns a `*mut c_void` pointing to its internal
    /// [`Slimmable<T>`], which can be conveniently passed over FFI boundaries
    /// that expect a pointer type, and used later with
    /// [`from_raw`][SlimBox::from_raw], [`SlimRef::from_raw`] or
    /// [`SlimMut::from_raw`].
    pub fn into_raw(self) -> *mut c_void {
        let ptr = self.as_raw();
        mem::forget(self);
        ptr
    }

    /// Reconstructs a `SlimBox` out of a `*mut c_void` that's pointing to a
    /// [`Slimmable<T>`].
    ///
    /// # Safety
    ///
    /// `pointer` must be an owned pointer to a valid `Slimmable<T>`, like the
    /// one returned by [`into_raw`][SlimBox::into_raw].
    pub unsafe fn from_raw(ptr: *mut c_void) -> SlimBox<T> {
        SlimBox {
            slim_box: SlimPtr::from_raw_unchecked(ptr),
            _phantom: PhantomData,
        }
    }

    /// Returns a [`SlimRef`] that borrows from this `SlimBox`.
    pub fn slimborrow(&self) -> SlimRef<'_, T> {
        SlimRef {
            slim_ref: self.slim_box,
            _phantom: PhantomData,
        }
    }

    /// Returns a [`SlimMut`] that borrows from this `SlimBox`.
    pub fn slimborrow_mut(&mut self) -> SlimMut<'_, T> {
        SlimMut {
            slim_mut: self.slim_box,
            _phantom: PhantomData,
        }
    }
}

/// `slimbox_unsize!(T, expression)` will unsize `expression` into a
/// [`SlimBox<T>`]. The `T` parameter can be omitted if it can be inferred.
#[macro_export]
macro_rules! slimbox_unsize {
    // load-bearing order, this arm must precede the (ty, expr) arm
    ($expression:expr) => {
        $crate::slimbox_unsize!(_, $expression)
    };
    ($T:ty, $expression:expr) => {
        $crate::SlimBox::<$T>::from_boxed_slimmable($crate::Slimmable::<$T, _>::boxed($expression))
    };
}

impl<T: ?Sized> Drop for SlimBox<T> {
    fn drop(&mut self) {
        // SAFETY: self.slim_box points to a valid Slimmable<T>
        let ptr = unsafe { self.slim_box.as_ptr() };
        // SAFETY: we were originally a Box<Slimmable<T>>
        unsafe { Box::from_raw(ptr) };
    }
}

impl<T: ?Sized> Deref for SlimBox<T> {
    type Target = T;

    fn deref(&self) -> &T {
        // SAFETY: self.slim_box points to a valid owned Slimmable<T>
        &unsafe { self.slim_box.as_ref() }.value
    }
}

impl<T: ?Sized> DerefMut for SlimBox<T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: self.slim_box points to a valid owned Slimmable<T>
        &mut unsafe { self.slim_box.as_mut() }.value
    }
}

/// A thin shared reference; like `&T` but guaranteed to be as big as a regular
/// pointer, even for `!Sized` types. It will typically refer to a value owned
/// by a [`SlimBox`], but it can also refer to a stack-allocated [`Slimmable`]
/// via [`from_slimmable`][SlimRef::from_slimmable].
#[derive(Debug)]
#[repr(transparent)]
pub struct SlimRef<'a, T: ?Sized> {
    /// Shared thin pointer to a `Slimmable`.
    slim_ref: SlimPtr<T>,
    /// Marker to act as if we have a `&T`.
    _phantom: PhantomData<&'a T>,
}
impl_unbound_copy! { SlimRef<'_, T> }

impl<T: ?Sized> SlimRef<'_, T> {
    /// Returns a `*const c_void` pointing to the internal [`Slimmable<T>`],
    /// which can be conveniently passed over FFI boundaries that expect a
    /// pointer type, and used later with [`from_raw`][SlimRef::from_raw].
    pub fn as_raw(&self) -> *const c_void {
        self.slim_ref.0.as_ptr() as *const c_void
    }

    /// Reconstructs a `SlimRef` out of a `*const c_void` that's pointing to a
    /// [`Slimmable<T>`].
    ///
    /// # Safety
    ///
    /// `ptr` must be a shared pointer to a valid `Slimmable<T>`, like the one
    /// returned by [`as_raw`][SlimRef::as_raw], [`SlimBox::into_raw`],
    /// [`SlimBox::as_raw`], or [`SlimMut::as_raw`].
    pub unsafe fn from_raw<'a>(ptr: *const c_void) -> SlimRef<'a, T> {
        SlimRef {
            slim_ref: SlimPtr::from_raw_unchecked(ptr as *mut c_void),
            _phantom: PhantomData,
        }
    }

    /// Returns a `SlimRef` that borrows the user-provided `slimmable`.
    pub fn from_slimmable(slimmable: &mut Slimmable<T>) -> SlimRef<'_, T> {
        let ptr = slimmable as *mut Slimmable<T>;

        // SAFETY: ptr points to a Slimmable<T>
        unsafe { &mut (*ptr).anchor }.write(SlimAnchor::new(ptr));

        // the following cast is sound because Slimmable<T> is repr(C) so a
        // pointer to it can be cast to a pointer to its anchor
        SlimRef {
            slim_ref: SlimPtr::new(ptr).unwrap(),
            _phantom: PhantomData,
        }
    }
}

impl<T: ?Sized> Deref for SlimRef<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        // SAFETY: self.slim_ref points to a valid shared Slimmable<T>
        &unsafe { self.slim_ref.as_ref() }.value
    }
}

/// A thin exclusive reference; like `&mut T` but guaranteed to be as big as a
/// regular pointer, even for `!Sized` types. It will typically refer to a value
/// owned by a [`SlimBox`], but it can also refer to a stack-allocated
/// [`Slimmable`] via [`from_slimmable`][SlimMut::from_slimmable].
#[derive(Debug)]
#[repr(transparent)]
pub struct SlimMut<'a, T: ?Sized> {
    /// Exclusive thin pointer to a `Slimmable`.
    slim_mut: SlimPtr<T>,
    /// Marker to act as if we have a `&mut T`.
    _phantom: PhantomData<&'a mut T>,
}

impl<T: ?Sized> SlimMut<'_, T> {
    /// Returns a `*mut c_void` pointing to the internal [`Slimmable<T>`], which
    /// can be conveniently passed over FFI boundaries that expect a pointer
    /// type, and used later with [`from_raw`][SlimMut::from_raw] or
    /// [`SlimRef::from_raw`].
    pub fn as_raw(&self) -> *mut c_void {
        self.slim_mut.0.as_ptr() as *mut c_void
    }

    /// Reconstructs a `SlimMut` out of a `*mut c_void` that's pointing to a
    /// [`Slimmable<T>`].
    ///
    /// # Safety
    ///
    /// `ptr` must be an exclusive pointer to a valid `Slimmable<T>`, like
    /// the one returned by [`as_raw`][SlimMut::as_raw], [`SlimBox::into_raw`], or
    /// [`SlimBox::as_raw`].
    pub unsafe fn from_raw<'a>(ptr: *mut c_void) -> SlimMut<'a, T> {
        SlimMut {
            slim_mut: SlimPtr::from_raw_unchecked(ptr),
            _phantom: PhantomData,
        }
    }

    /// Returns a [`SlimRef`] that reborrows from the current `SlimMut`.
    pub fn reborrow(&self) -> SlimRef<'_, T> {
        SlimRef {
            slim_ref: self.slim_mut,
            _phantom: PhantomData,
        }
    }

    /// Returns a new `SlimMut` that reborrows from the current one.
    pub fn reborrow_mut(&mut self) -> SlimMut<'_, T> {
        SlimMut {
            slim_mut: self.slim_mut,
            _phantom: PhantomData,
        }
    }

    /// Returns a `SlimMut` that borrows the user-provided `slimmable`.
    pub fn from_slimmable(slimmable: &mut Slimmable<T>) -> SlimMut<'_, T> {
        let ptr = slimmable as *mut Slimmable<T>;

        // SAFETY: ptr points to a Slimmable<T>
        unsafe { &mut (*ptr).anchor }.write(SlimAnchor::new(ptr));

        // the following cast is sound because Slimmable<T> is repr(C) so a
        // pointer to it can be cast to a pointer to its anchor
        SlimMut {
            slim_mut: SlimPtr::new(ptr).unwrap(),
            _phantom: PhantomData,
        }
    }
}

impl<T: ?Sized> Deref for SlimMut<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        // SAFETY: self.slim_mut points to a valid exclusive Slimmable<T>
        &unsafe { self.slim_mut.as_ref() }.value
    }
}

impl<T: ?Sized> DerefMut for SlimMut<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: self.slim_mut points to a valid exclusive Slimmable<T>
        &mut unsafe { self.slim_mut.as_mut() }.value
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn sized() {
        let mut foo = SlimBox::new(42u64);

        assert_eq!(*foo, 42);

        let foo_ref = foo.slimborrow();
        assert_eq!(*foo_ref, 42);

        let mut foo_mut = foo.slimborrow_mut();
        *foo_mut = 420;

        let mut foo_mut_2 = foo_mut.reborrow_mut();
        *foo_mut_2 = 42;

        *foo_mut = 420;

        let foo_ref = foo.slimborrow();
        let foo_ref2 = foo.slimborrow();

        assert_eq!(*foo_ref, 420);

        assert_eq!(*foo, 420);
        assert_eq!(*foo_ref2, 420);
    }

    #[test]
    #[cfg(any(feature = "unsafe_stable", feature = "nightly"))]
    fn slice() {
        let foo = SlimBox::<[i32]>::from_box(Box::new([1, 2, 3, 4]));

        assert_eq!(mem::size_of_val(&foo), mem::size_of::<*const ()>());

        assert_eq!(
            mem::size_of_val(&foo.deref()),
            mem::size_of::<*const [i32]>(),
        );

        assert_eq!(foo.len(), 4);
        assert_eq!(foo[1], 2);

        let bar = SlimBox::<[i32]>::from_box(Box::new([1, 2, 3, 4, 5]));
        assert_eq!(bar.len(), 5);

        use alloc::{sync::Arc, vec::Vec};

        let sentinel = Arc::new(());
        let foo = SlimBox::<[Arc<()>]>::from_box(Box::new([
            sentinel.clone(),
            sentinel.clone(),
            sentinel.clone(),
        ]));

        drop(foo);
        Arc::try_unwrap(sentinel).unwrap();

        let sentinel = Arc::new(());
        let foo =
            SlimBox::from_box(Vec::from([sentinel.clone(), sentinel.clone()]).into_boxed_slice());
        drop(foo);
        Arc::try_unwrap(sentinel).unwrap();
    }

    #[test]
    #[cfg(any(feature = "unsafe_stable", feature = "nightly"))]
    fn fn_trait() {
        let y = 5;

        let foo_box: Box<dyn Fn() -> i32> = Box::new(|| y);
        let foo: SlimBox<dyn Fn() -> i32> = SlimBox::from_box(Box::new(|| y));

        assert_eq!(mem::size_of_val(&foo), mem::size_of::<*const ()>());

        assert_eq!(
            mem::size_of_val(&foo_box),
            mem::size_of::<*const dyn Fn() -> i32>()
        );

        assert_eq!(foo_box(), 5);
        assert_eq!(foo(), 5);

        let foo_from_box = SlimBox::from_box(foo_box);
        assert_eq!(mem::size_of_val(&foo_from_box), mem::size_of::<*const ()>());
        assert_eq!(foo_from_box(), 5);
    }

    #[test]
    fn ffi() {
        extern "C" fn calls_callback(
            cb: extern "C" fn(*mut c_void) -> i32,
            userdata: *mut c_void,
        ) -> i32 {
            cb(userdata)
        }

        extern "C" fn trampoline(userdata: *mut c_void) -> i32 {
            let userdata: SlimRef<dyn Fn() -> i32> = unsafe { SlimRef::from_raw(userdata) };
            userdata()
        }

        let six_times_nine = slimbox_unsize!(dyn Fn() -> i32, || 42);

        let r = calls_callback(
            trampoline,
            six_times_nine.slimborrow().as_raw() as *mut c_void,
        );

        assert_eq!(r, 42);
    }

    #[test]
    fn stack() {
        use alloc::string::ToString;

        let mut b = Slimmable::new(42);
        let r: SlimMut<dyn ToString> = SlimMut::from_slimmable(&mut b);
        assert_eq!(r.to_string(), "42");
    }
}
