#![feature(ptr_metadata)]
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
    ptr::{self, NonNull, Pointee},
};

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

unsafe fn dealloc(ptr: *mut u8, layout: Layout) {
    if layout.size() != 0 {
        alloc::alloc::dealloc(ptr, layout);
    }
}

type Metadata<T> = <T as Pointee>::Metadata;

#[repr(C)]
struct Inner<M: ?Sized, V: ?Sized = M> {
    metadata: Metadata<M>,
    value: V,
}

#[derive(Debug)]
#[repr(transparent)]
struct InnerPtr<T: ?Sized>(NonNull<Metadata<T>>);

impl<T: ?Sized> Copy for InnerPtr<T> {}
impl<T: ?Sized> Clone for InnerPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized> InnerPtr<T> {
    fn from_box(boxed: Box<Inner<T>>) -> Self {
        Self(NonNull::new(Box::into_raw(boxed) as *mut Metadata<T>).unwrap())
    }

    unsafe fn as_ptr(self) -> *mut Inner<T> {
        let metadata = *self.0.as_ptr();
        ptr::from_raw_parts_mut::<T>(self.0.as_ptr().cast(), metadata) as _
    }

    unsafe fn as_ref<'a>(self) -> &'a Inner<T> {
        &*self.as_ptr()
    }

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
        let metadata = ptr::metadata(&value);
        let boxed = Box::new(Inner { metadata, value });

        Self {
            inner_box: InnerPtr::from_box(boxed),
            _phantom: PhantomData,
        }
    }

    #[cfg(feature = "unsize")]
    pub fn from_unsize<S: core::marker::Unsize<T>>(value: S) -> Self {
        let metadata = ptr::metadata(&value as &T);
        let boxed = Box::new(Inner { metadata, value });

        Self {
            inner_box: InnerPtr::from_box(boxed),
            _phantom: PhantomData,
        }
    }

    pub fn from_box(boxed: Box<T>) -> Self {
        let metadata = ptr::metadata(&*boxed);

        // we manually build the Layout for an Inner<T> that can hold the
        // currently boxed value
        let inner_layout = Layout::from_size_align(0, 1).unwrap();
        let metadata_layout = Layout::for_value(&metadata);
        let (inner_layout, metadata_offset) = inner_layout.extend(metadata_layout).unwrap();
        let value_layout = Layout::for_value(&*boxed);
        let (inner_layout, value_offset) = inner_layout.extend(value_layout).unwrap();
        let inner_layout = inner_layout.pad_to_align();

        let inner_storage = alloc(inner_layout);

        let value_storage = Box::into_raw(boxed);

        // SAFETY: we're initializing the newly-allocated Inner<T> that lives at
        // inner_storage, copying metadata and moving the value in value_storage
        unsafe {
            ptr::write(inner_storage.add(metadata_offset).cast(), metadata);
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
