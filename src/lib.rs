#![cfg_attr(feature = "unsize", feature(unsize))]
#![no_std]
extern crate alloc;

use alloc::boxed::Box;
use core::{
    alloc::Layout,
    ffi::c_void,
    marker::PhantomData,
    mem,
    ops::{Deref, DerefMut},
    ptr::{self, NonNull},
};

#[cfg(feature = "unsize")]
use core::marker::Unsize;

/// Stable version of [`pointer::set_ptr_value`][set_ptr_value] that does enough
/// checks to hopefully make it loudly fail if and when the layout of `*mut T`
/// for `T: !Sized` changes.
///
/// [set_ptr_value]: https://doc.rust-lang.org/std/primitive.pointer.html#method.set_ptr_value
fn set_ptr_value<T: ?Sized>(ptr: *mut T, val: *mut u8) -> *mut T {
    assert!(mem::size_of::<*mut T>() >= mem::size_of::<*mut u8>());
    assert!(mem::align_of::<*mut T>() == mem::align_of::<*mut u8>());

    let mut new_ptr = ptr;
    let thin = &mut new_ptr as *mut *mut T as *mut *mut u8;

    unsafe {
        thin.write((ptr as *mut u8).wrapping_add(1));
        assert_eq!(new_ptr as *mut u8, (ptr as *mut u8).wrapping_add(1));

        thin.write(val);
        assert_eq!(new_ptr as *mut u8, val);
    };

    new_ptr
}

/// [`alloc::alloc::alloc`] but returns a dangling aligned pointer on a
/// zero-sized allocation.
fn alloc(layout: Layout) -> *mut u8 {
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
unsafe fn dealloc(ptr: *mut u8, layout: Layout) {
    if layout.size() != 0 {
        alloc::alloc::dealloc(ptr, layout);
    }
}

/// A container for a potentially wide pointer to the container, and a value.
///
/// Invariant: the `this` pointer points to (the unsizing of) the struct itself,
/// so the struct should never be exposed to user code that could move it.
#[repr(C)]
struct Inner<M: ?Sized, V: ?Sized = M> {
    this: *mut Inner<M>,
    value: V,
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
    /// `self` must point to a valid `Inner<T>`.
    unsafe fn as_ptr(self) -> *mut Inner<T> {
        *self.0.as_ptr()
    }

    /// Returns a shared reference to the pointed [`Inner<T>`], with arbitrary
    /// lifetime.
    ///
    /// # Safety
    ///
    /// `self` must point to a valid `Inner<T>`.
    unsafe fn as_ref<'a>(self) -> &'a Inner<T> {
        &*self.as_ptr()
    }

    /// Returns an exclusive reference to the pointed [`Inner<T>`], with
    /// arbitrary lifetime.
    ///
    /// # Safety
    ///
    /// `self` must point to a valid `Inner<T>`.
    unsafe fn as_mut<'a>(self) -> &'a mut Inner<T> {
        &mut *self.as_ptr()
    }
}

#[derive(Debug)]
#[repr(transparent)]
pub struct SlimBox<T: ?Sized> {
    inner_box: InnerPtr<T>,
    _phantom: PhantomData<Box<Inner<T>>>,
}

impl<T: ?Sized> SlimBox<T> {
    pub fn new(value: T) -> Self
    where
        T: Sized,
    {
        let this = &value as &T as *const T as *mut Inner<T>;

        let mut boxed = Box::new(Inner { this, value });
        boxed.this = &*boxed as *const Inner<T> as *mut Inner<T>;

        Self {
            inner_box: InnerPtr(NonNull::new(Box::into_raw(boxed)).unwrap().cast()),
            _phantom: PhantomData,
        }
    }

    #[cfg(feature = "unsize")]
    pub fn from_unsize<S: Unsize<T>>(value: S) -> Self {
        let this = &value as &T as *const T as *mut Inner<T>;

        let mut boxed = Box::new(Inner { this, value });
        boxed.this = &*boxed as *const Inner<T> as *mut Inner<T>;

        Self {
            inner_box: InnerPtr(NonNull::new(Box::into_raw(boxed)).unwrap().cast()),
            _phantom: PhantomData,
        }
    }

    /// # Safety
    ///
    /// The metadata in `ptr` must be valid metadata for a pointer to `value` as T.
    pub unsafe fn from_unsize_and_ptr<S>(value: S, ptr: *const T) -> Self {
        let this = ptr as *mut Inner<T>;

        let mut boxed = Box::new(Inner::<T, S> { this, value });
        boxed.this = set_ptr_value(this, &*boxed as *const Inner<T, S> as *mut u8);

        Self {
            inner_box: InnerPtr(NonNull::new(Box::into_raw(boxed)).unwrap().cast()),
            _phantom: PhantomData,
        }
    }

    pub fn from_box(boxed: Box<T>) -> Self {
        // we manually build the Layout for an Inner<T> that can hold the
        // currently boxed value
        let inner_layout = Layout::new::<*mut Inner<T>>();
        let value_layout = Layout::for_value(&*boxed);
        let (inner_layout, value_offset) = inner_layout.extend(value_layout).unwrap();
        let inner_layout = inner_layout.pad_to_align();

        let inner_storage = alloc(inner_layout);

        let value_storage = Box::into_raw(boxed);

        // SAFETY: we're initializing the newly-allocated Inner<T> that lives at
        // inner_storage, copying metadata and moving the value in value_storage
        unsafe {
            ptr::write(
                inner_storage.cast(),
                set_ptr_value(value_storage, inner_storage) as *mut Inner<T>,
            );
            ptr::copy_nonoverlapping(
                value_storage.cast(),
                inner_storage.add(value_offset),
                value_layout.size(),
            );
        }

        // SAFETY: value_storage is an allocated T that we moved from
        unsafe { dealloc(value_storage.cast(), value_layout) };

        Self {
            inner_box: InnerPtr(NonNull::new(inner_storage).unwrap().cast()),
            _phantom: PhantomData,
        }
    }

    fn inner_ref(&self) -> &Inner<T> {
        unsafe { self.inner_box.as_ref() }
    }

    fn inner_mut(&mut self) -> &mut Inner<T> {
        unsafe { self.inner_box.as_mut() }
    }

    pub fn into_raw(self) -> *mut c_void {
        let pointer = self.inner_box.0.as_ptr() as _;
        mem::forget(self);
        pointer
    }

    /// # Safety
    ///
    /// `pointer` must be an owned pointer to the inner value of a [`SlimBox`],
    /// like the one returned by [`into_raw`][Self::into_raw].
    pub unsafe fn from_raw(pointer: *mut c_void) -> Self {
        Self {
            inner_box: InnerPtr(NonNull::new_unchecked(pointer).cast()),
            _phantom: PhantomData,
        }
    }
}

#[macro_export]
macro_rules! slimbox_unsize {
    ($expression:expr) => {{
        let val = $expression;
        // inference assuming this expression is known to be a SlimBox<T>: `ptr`
        // is a *const T and the first conversion from a &_ must be to its
        // *const
        let ptr = &val as &_ as *const _;
        unsafe { $crate::SlimBox::from_unsize_and_ptr(val, ptr) }
    }};
    ($T:ty, $expression:expr) => {{
        let val = $expression;
        let ptr = &val as &$T as *const $T;
        unsafe { $crate::SlimBox::<$T>::from_unsize_and_ptr(val, ptr) }
    }};
}

impl<T: ?Sized> Drop for SlimBox<T> {
    fn drop(&mut self) {
        unsafe { Box::from_raw(self.inner_box.as_ptr()) };
    }
}

impl<T: ?Sized> Deref for SlimBox<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.inner_ref().value
    }
}

impl<T: ?Sized> DerefMut for SlimBox<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.inner_mut().value
    }
}

// SAFETY: same bounds as Box<T>
unsafe impl<T: ?Sized + Send> Send for SlimBox<T> {}
unsafe impl<T: ?Sized + Sync> Sync for SlimBox<T> {}

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
    fn inner_ref(&self) -> &Inner<T> {
        unsafe { self.inner_ref.as_ref() }
    }

    pub fn as_raw(self) -> *const c_void {
        self.inner_ref.0.as_ptr() as _
    }

    /// # Safety
    ///
    /// `pointer` must be a shared pointer to the inner value of a [`SlimBox`],
    /// like the one returned by [`as_raw`][Self::as_raw], [`SlimBox::into_raw`]
    /// or [`SlimMut::as_raw`].
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
        &self.inner_ref().value
    }
}

// SAFETY: same bounds as &T
unsafe impl<T: ?Sized + Sync> Send for SlimRef<'_, T> {}
unsafe impl<T: ?Sized + Sync> Sync for SlimRef<'_, T> {}

#[derive(Debug)]
#[repr(transparent)]
pub struct SlimMut<'a, T: ?Sized> {
    inner_mut: InnerPtr<T>,
    _phantom: PhantomData<&'a mut Inner<T>>,
}

impl<T: ?Sized> SlimMut<'_, T> {
    fn inner_ref(&self) -> &Inner<T> {
        unsafe { self.inner_mut.as_ref() }
    }

    fn inner_mut(&mut self) -> &mut Inner<T> {
        unsafe { self.inner_mut.as_mut() }
    }

    pub fn as_raw(self) -> *mut c_void {
        self.inner_mut.0.as_ptr() as _
    }

    /// # Safety
    ///
    /// `pointer` must be an exclusive pointer to the inner value of a
    /// [`SlimBox`], like the one returned by [`as_raw`][Self::as_raw] or
    /// [`SlimBox::into_raw`].
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
        &self.inner_ref().value
    }
}

impl<T: ?Sized> DerefMut for SlimMut<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.inner_mut().value
    }
}

// SAFETY: same bounds as &mut T
unsafe impl<T: ?Sized + Send> Send for SlimMut<'_, T> {}
unsafe impl<T: ?Sized + Sync> Sync for SlimMut<'_, T> {}

pub trait AsSlimRef<T: ?Sized> {
    fn as_slim_ref(&self) -> SlimRef<'_, T>;
}
pub trait AsSlimMut<T: ?Sized> {
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

    use core::mem;

    #[test]
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

        let six_times_nine: SlimBox<dyn Fn() -> i32> = SlimBox::from_box(Box::new(|| 42));

        let r = calls_callback(
            trampoline,
            six_times_nine.as_slim_ref().as_raw() as *mut c_void,
        );

        assert_eq!(r, 42);
    }

    #[cfg(feature = "unsize")]
    #[test]
    fn unsize() {
        let foo = SlimBox::<[i32]>::from_unsize([1, 2, 3, 4]);

        assert_eq!(mem::size_of_val(&foo), mem::size_of::<*const ()>());

        assert_eq!(
            mem::size_of_val(&foo.deref()),
            mem::size_of::<*const [i32]>(),
        );

        assert_eq!(foo.len(), 4);
        assert_eq!(foo[1], 2);
    }
}