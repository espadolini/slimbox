#![cfg_attr(feature = "nightly", feature(set_ptr_value))]
#![no_std]
extern crate alloc;

use alloc::boxed::Box;
use core::{
    ffi::c_void,
    marker::PhantomData,
    mem,
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
/// [set_ptr_value]: https://doc.rust-lang.org/std/primitive.pointer.html#method.set_ptr_value
#[cfg(all(feature = "unsound_stable", not(feature = "nightly")))]
fn set_ptr_value<T: ?Sized>(mut ptr: *mut T, val: *mut u8) -> *mut T {
    assert!(mem::size_of::<*mut T>() >= mem::size_of::<*mut u8>());
    assert!(mem::align_of::<*mut T>() == mem::align_of::<*mut u8>());

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
#[cfg(any(feature = "unsound_stable", feature = "nightly"))]
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
#[cfg(any(feature = "unsound_stable", feature = "nightly"))]
unsafe fn dealloc_extended(ptr: *mut u8, layout: core::alloc::Layout) {
    if layout.size() != 0 {
        alloc::alloc::dealloc(ptr, layout);
    }
}

/// A container for a potentially wide pointer to the unsized container, and a
/// value. The internal allocation type used by [`SlimBox`].
///
/// Intended for internal use.
#[doc(hidden)]
#[repr(C)]
pub struct Inner<T: ?Sized, S: ?Sized = T> {
    this: Option<NonNull<Inner<T>>>,
    value: S,
}

impl<T: ?Sized, S> Inner<T, S> {
    /// Make an `Box<Inner<T, S>>` out of a `value` of type `S`.
    ///
    /// Intended for internal use.
    #[doc(hidden)]
    pub fn boxed(value: S) -> Box<Self> {
        Box::new(Self { this: None, value })
    }
}

/// A non-null thin pointer to [`Inner<T>`], by virtue of pointing to its first
/// field.
#[derive(Debug)]
#[repr(transparent)]
struct InnerPtr<T: ?Sized>(NonNull<*mut Inner<T>>);

// manual impl to avoid bounds on T
impl<T: ?Sized> Copy for InnerPtr<T> {}
impl<T: ?Sized> Clone for InnerPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized> InnerPtr<T> {
    /// Returns a regular (potentially wide) pointer to the pointed
    /// [`Inner<T>`].
    ///
    /// # Safety
    ///
    /// `self` must point to a valid `Inner<T>` with a valid `this` member.
    unsafe fn as_ptr(self) -> *mut Inner<T> {
        *self.0.as_ptr()
    }

    /// Returns a shared reference to the pointed [`Inner<T>`], with arbitrary
    /// lifetime.
    ///
    /// # Safety
    ///
    /// `self` must point to a valid `Inner<T>` with a valid `this` member and
    /// it must be legal to build a shared reference to it.
    unsafe fn as_ref<'a>(self) -> &'a Inner<T> {
        &*self.as_ptr()
    }

    /// Returns an exclusive reference to the pointed [`Inner<T>`], with
    /// arbitrary lifetime.
    ///
    /// # Safety
    ///
    /// `self` must point to a valid `Inner<T>` with a valid `this` member and
    /// it must be legal to build an exclusive reference to it.
    unsafe fn as_mut<'a>(self) -> &'a mut Inner<T> {
        &mut *self.as_ptr()
    }
}

/// A thin owned pointer; like [`Box<T>`] but guaranteed to be as big as a
/// regular pointer, even for `!Sized` types.
///
/// Internally this points to a single allocation that contains a (potentially)
/// wide pointer, and the value itself.
#[derive(Debug)]
#[repr(transparent)]
pub struct SlimBox<T: ?Sized> {
    inner_box: InnerPtr<T>,
    _phantom: PhantomData<Box<Inner<T>>>,
}

impl<T: ?Sized> SlimBox<T> {
    /// Repacks a `Box<Inner<T>>` into a `SlimBox<T>`, thinning its pointer.
    ///
    /// Intended for internal use.
    #[doc(hidden)]
    pub fn from_boxed_inner(boxed: Box<Inner<T>>) -> Self {
        let boxed = Box::into_raw(boxed);

        // SAFETY: we just unpacked `boxed` from a `Box<Inner<T>>`
        unsafe { (*boxed).this = Some(NonNull::new_unchecked(boxed)) };

        // the following cast is sound because Inner<T> is repr(C) so a pointer
        // can be cast to its first field, which is a Some of
        // Option<NonNull<Inner<T>>> which is compatible with *mut Inner<T>
        Self {
            inner_box: InnerPtr(NonNull::new(boxed as *mut *mut Inner<T>).unwrap()),
            _phantom: PhantomData,
        }
    }

    /// Moves `value` in a new `SlimBox<T>`. `T` must be `Sized` to use this.
    pub fn new(value: T) -> Self
    where
        T: Sized,
    {
        Self::from_boxed_inner(Inner::boxed(value))
    }

    /// Moves the value contained in `boxed` into a `SlimBox`. This function
    /// makes a new allocation.
    #[cfg(any(feature = "unsound_stable", feature = "nightly"))]
    pub fn from_box(boxed: Box<T>) -> Self {
        use core::alloc::Layout;

        // we manually build the Layout for an Inner<T> that can hold the
        // currently boxed value
        let inner_layout = Layout::new::<*mut Inner<T>>();
        let value_layout = Layout::for_value(&*boxed);
        let (inner_layout, value_offset) = inner_layout.extend(value_layout).unwrap();
        let inner_layout = inner_layout.pad_to_align();

        let inner_ptr = alloc_extended(inner_layout);
        let value_ptr = Box::into_raw(boxed);

        // SAFETY: we're initializing the newly-allocated Inner<T> that lives at
        // inner_ptr, moving the T from the allocation at value_ptr
        unsafe {
            core::ptr::write(inner_ptr as *mut Option<NonNull<Inner<T>>>, None);

            core::ptr::copy_nonoverlapping(
                value_ptr as *const u8,
                inner_ptr.add(value_offset),
                value_layout.size(),
            );
            dealloc_extended(value_ptr as *mut u8, value_layout);
        }

        // copy the T metadata from the boxed pointer
        let inner_ptr = set_ptr_value(value_ptr, inner_ptr) as *mut Inner<T>;

        // SAFETY: inner_ptr points to a valid, initialized, uniquely owned Inner<T>
        Self::from_boxed_inner(unsafe { Box::from_raw(inner_ptr) })
    }

    /// Returns a `*mut c_void` pointing to the internal allocation, which can
    /// be conveniently passed over FFI boundaries that expect a pointer type,
    /// and used later with [`SlimRef::from_raw`] or [`SlimMut::from_raw`].
    pub fn as_raw(&self) -> *mut c_void {
        self.inner_box.0.as_ptr() as _
    }

    /// Consumes the box and returns a `*mut c_void` pointing to its internal
    /// allocation, which can be conveniently passed over FFI boundaries that
    /// expect a pointer type, and used later with [`from_raw`][Self::from_raw],
    /// [`SlimRef::from_raw`] or [`SlimMut::from_raw`].
    pub fn into_raw(self) -> *mut c_void {
        let pointer = self.as_raw();
        mem::forget(self);
        pointer
    }

    /// Reconstructs a `SlimBox` out of a `*mut c_void` that's pointing to an
    /// internal allocation of the correct type.
    ///
    /// # Safety
    ///
    /// `pointer` must be an owned pointer to the inner value of a [`SlimBox`],
    /// like the one returned by [`into_raw`][Self::into_raw].
    pub unsafe fn from_raw(pointer: *mut c_void) -> Self {
        Self {
            inner_box: InnerPtr(NonNull::new_unchecked(pointer as *mut *mut Inner<T>)),
            _phantom: PhantomData,
        }
    }
}

/// `slimbox_unsize!(T, expression)` will unsize `expression` into a
/// [`SlimBox<T>`]. The `T` parameter can be omitted if it can be inferred.
#[macro_export]
macro_rules! slimbox_unsize {
    ($expression:expr) => {
        $crate::SlimBox::from_boxed_inner($crate::Inner::boxed($expression))
    };
    ($T:ty, $expression:expr) => {
        $crate::SlimBox::<$T>::from_boxed_inner($crate::Inner::<$T, _>::boxed($expression))
    };
}

impl<T: ?Sized> Drop for SlimBox<T> {
    fn drop(&mut self) {
        unsafe { Box::from_raw(self.inner_box.as_ptr()) };
    }
}

impl<T: ?Sized> Deref for SlimBox<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &unsafe { self.inner_box.as_ref() }.value
    }
}

impl<T: ?Sized> DerefMut for SlimBox<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut unsafe { self.inner_box.as_mut() }.value
    }
}

// SAFETY: same bounds as Box<T>
unsafe impl<T: ?Sized + Send> Send for SlimBox<T> {}
unsafe impl<T: ?Sized + Sync> Sync for SlimBox<T> {}

/// A thin shared reference; like `&T` but guaranteed to be as big as a regular
/// pointer, even for `!Sized` types. It will typically refer to a value owned
/// by a [`SlimBox`].
///
/// [reference]: https://doc.rust-lang.org/std/primitive.reference.html
#[derive(Debug)]
#[repr(transparent)]
pub struct SlimRef<'a, T: ?Sized> {
    inner_ref: InnerPtr<T>,
    _phantom: PhantomData<&'a Inner<T>>,
}

// manual impl to avoid bounds on T
impl<T: ?Sized> Copy for SlimRef<'_, T> {}
impl<T: ?Sized> Clone for SlimRef<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized> SlimRef<'_, T> {
    /// Returns a `*const c_void` pointing to the internal allocation, which can
    /// be conveniently passed over FFI boundaries that expect a pointer type,
    /// and used later with [`from_raw`][Self::from_raw].
    pub fn as_raw(&self) -> *const c_void {
        self.inner_ref.0.as_ptr() as _
    }

    /// Reconstructs a `SlimRef` out of a `*const c_void` that's pointing to an
    /// internal allocation of the correct type.
    ///
    /// # Safety
    ///
    /// `pointer` must be a shared pointer to the inner value of a [`SlimBox`],
    /// like the one returned by [`as_raw`][Self::as_raw],
    /// [`SlimBox::into_raw`], [`SlimBox::as_raw`], or [`SlimMut::as_raw`].
    pub unsafe fn from_raw(pointer: *const c_void) -> Self {
        Self {
            inner_ref: InnerPtr(NonNull::new_unchecked(pointer as *mut c_void).cast()),
            _phantom: PhantomData,
        }
    }
}

impl<T: ?Sized> Deref for SlimRef<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        &unsafe { self.inner_ref.as_ref() }.value
    }
}

// SAFETY: same bounds as &T
unsafe impl<T: ?Sized + Sync> Send for SlimRef<'_, T> {}
unsafe impl<T: ?Sized + Sync> Sync for SlimRef<'_, T> {}

/// A thin exclusive reference; like `&mut T` but guaranteed to be as big as a
/// regular pointer, even for `!Sized` types. It will typically refer to a value
/// owned by a [`SlimBox`].
#[derive(Debug)]
#[repr(transparent)]
pub struct SlimMut<'a, T: ?Sized> {
    inner_mut: InnerPtr<T>,
    _phantom: PhantomData<&'a mut Inner<T>>,
}

impl<T: ?Sized> SlimMut<'_, T> {
    /// Returns a `*mut c_void` pointing to the internal allocation, which can
    /// be conveniently passed over FFI boundaries that expect a pointer type,
    /// and used later with [`from_raw`][Self::from_raw] or
    /// [`SlimRef::from_raw`].
    pub fn as_raw(&self) -> *mut c_void {
        self.inner_mut.0.as_ptr() as _
    }

    /// Reconstructs a `SlimMut` out of a `*mut c_void` that's pointing to an
    /// internal allocation of the correct type.
    ///
    /// # Safety
    ///
    /// `pointer` must be an exclusive pointer to the inner value of a
    /// [`SlimBox`], like the one returned by [`as_raw`][Self::as_raw],
    /// [`SlimBox::into_raw`], or [`SlimBox::as_raw`].
    pub unsafe fn from_raw(pointer: *mut c_void) -> Self {
        Self {
            inner_mut: InnerPtr(NonNull::new_unchecked(pointer).cast()),
            _phantom: PhantomData,
        }
    }
}

impl<T: ?Sized> Deref for SlimMut<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        &unsafe { self.inner_mut.as_ref() }.value
    }
}

impl<T: ?Sized> DerefMut for SlimMut<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut unsafe { self.inner_mut.as_mut() }.value
    }
}

// SAFETY: same bounds as &mut T
unsafe impl<T: ?Sized + Send> Send for SlimMut<'_, T> {}
unsafe impl<T: ?Sized + Sync> Sync for SlimMut<'_, T> {}

/// Trait for types that can be sharedly borrowed as a `SlimRef`.
pub trait AsSlimRef<T: ?Sized> {
    /// Returns a new `SlimRef` sharedly borrowing from `self`.
    fn as_slim_ref(&self) -> SlimRef<'_, T>;
}

/// Trait for types that can be exclusively borrowed as a `SlimMut`.
pub trait AsSlimMut<T: ?Sized> {
    /// Returns a new `SlimMut` exclusively borrowing from `self`.
    fn as_slim_mut(&mut self) -> SlimMut<'_, T>;
}

impl<T: ?Sized> AsSlimRef<T> for SlimBox<T> {
    fn as_slim_ref(&self) -> SlimRef<'_, T> {
        SlimRef {
            inner_ref: self.inner_box,
            _phantom: PhantomData,
        }
    }
}

impl<T: ?Sized> AsSlimMut<T> for SlimBox<T> {
    fn as_slim_mut(&mut self) -> SlimMut<'_, T> {
        SlimMut {
            inner_mut: self.inner_box,
            _phantom: PhantomData,
        }
    }
}

impl<T: ?Sized> AsSlimRef<T> for SlimRef<'_, T> {
    fn as_slim_ref(&self) -> SlimRef<'_, T> {
        SlimRef {
            inner_ref: self.inner_ref,
            _phantom: PhantomData,
        }
    }
}

impl<T: ?Sized> AsSlimRef<T> for SlimMut<'_, T> {
    fn as_slim_ref(&self) -> SlimRef<'_, T> {
        SlimRef {
            inner_ref: self.inner_mut,
            _phantom: PhantomData,
        }
    }
}

impl<T: ?Sized> AsSlimMut<T> for SlimMut<'_, T> {
    fn as_slim_mut(&mut self) -> SlimMut<'_, T> {
        SlimMut {
            inner_mut: self.inner_mut,
            _phantom: PhantomData,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn sized() {
        let mut foo = SlimBox::new(42u64);

        assert_eq!(*foo, 42);

        let foo_ref = foo.as_slim_ref();
        assert_eq!(*foo_ref, 42);

        let mut foo_mut = foo.as_slim_mut();
        *foo_mut = 420;

        let foo_ref = foo.as_slim_ref();
        let foo_ref2 = foo.as_slim_ref();

        assert_eq!(*foo_ref, 420);

        assert_eq!(*foo, 420);
        assert_eq!(*foo_ref2, 420);
    }

    #[test]
    #[cfg(any(feature = "unsound_stable", feature = "nightly"))]
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
    #[cfg(any(feature = "unsound_stable", feature = "nightly"))]
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
            six_times_nine.as_slim_ref().as_raw() as *mut c_void,
        );

        assert_eq!(r, 42);
    }
}
