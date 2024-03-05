//! # Keyed Set: a hashbrown-based HashSet that indexes based on projections of its elements.
//! Ever wanted a `HashMap<K, V>`, but where `V` actually contains `K` (or at least can be projected to it)?
//! Well this is it.
//!
//! The easiest way to define a projection is through a closure that you pass at construction, but you may also define your own key extractors as ZSTs that implement `Default` to gain a `Default` constructor for your Keyed Sets.

#![no_std]

use core::{
    hash::{BuildHasher, Hash, Hasher},
    marker::PhantomData,
};

use hashbrown::{
    hash_map::DefaultHashBuilder,
    raw::{RawIntoIter, RawIter, RawTable},
};

/// A `HashMap<K, V>` where `K` is a part of `V`
#[derive(Clone)]
pub struct KeyedSet<T, Extractor, S = DefaultHashBuilder> {
    inner: hashbrown::raw::RawTable<T>,
    hash_builder: S,
    extractor: Extractor,
}

impl<T, Extractor: Default, S: Default> Default for KeyedSet<T, Extractor, S> {
    fn default() -> Self {
        Self {
            inner: Default::default(),
            hash_builder: Default::default(),
            extractor: Default::default(),
        }
    }
}

impl<'a, T, Extractor, S> IntoIterator for &'a KeyedSet<T, Extractor, S> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
impl<'a, T, Extractor, S> IntoIterator for &'a mut KeyedSet<T, Extractor, S> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}
/// Extracts the key from the value, allowing [`KeyedSet`] to obtain its values' keys.
pub trait KeyExtractor<'a, T> {
    /// The type of the key extracted by the extractor.
    type Key: Hash;
    /// Extracts the key from the value, allowing [`KeyedSet`] to obtain its values' keys.
    fn extract(&self, from: &'a T) -> Self::Key;
}
impl<'a, T: 'a, U: Hash, F: Fn(&'a T) -> U> KeyExtractor<'a, T> for F {
    type Key = U;
    fn extract(&self, from: &'a T) -> Self::Key {
        self(from)
    }
}
impl<'a, T: 'a + Hash> KeyExtractor<'a, T> for () {
    type Key = &'a T;
    fn extract(&self, from: &'a T) -> Self::Key {
        from
    }
}
impl<T, Extractor> KeyedSet<T, Extractor>
where
    Extractor: for<'a> KeyExtractor<'a, T>,
    for<'a> <Extractor as KeyExtractor<'a, T>>::Key: core::hash::Hash,
{
    /// Construct a new map where the key is extracted from the value using `extractor`.`
    pub fn new(extractor: Extractor) -> Self {
        Self {
            inner: Default::default(),
            hash_builder: Default::default(),
            extractor,
        }
    }
}

impl<T: core::fmt::Debug, Extractor, S> core::fmt::Debug for KeyedSet<T, Extractor, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "KeyedSet {{")?;
        for v in self.iter() {
            write!(f, "{:?}, ", v)?;
        }
        write!(f, "}}")
    }
}

#[allow(clippy::manual_hash_one)]
impl<T, Extractor, S> KeyedSet<T, Extractor, S>
where
    Extractor: for<'a> KeyExtractor<'a, T>,
    for<'a> <Extractor as KeyExtractor<'a, T>>::Key: core::hash::Hash,
    S: BuildHasher,
{
    /// Inserts a value into the map.
    pub fn insert(&mut self, value: T) -> Option<T>
    where
        for<'a, 'b> <Extractor as KeyExtractor<'a, T>>::Key:
            PartialEq<<Extractor as KeyExtractor<'b, T>>::Key>,
    {
        let key = self.extractor.extract(&value);
        let mut hasher = self.hash_builder.build_hasher();
        key.hash(&mut hasher);
        let hash = hasher.finish();
        match self
            .inner
            .get_mut(hash, |i| self.extractor.extract(i).eq(&key))
        {
            Some(bucket) => {
                core::mem::drop(key);
                Some(core::mem::replace(bucket, value))
            }
            None => {
                core::mem::drop(key);
                let hasher = make_hasher(&self.hash_builder, &self.extractor);
                self.inner.insert(hash, value, hasher);
                None
            }
        }
    }
    /// Obtain an entry in the map, allowing mutable access to the value associated to that key if it exists.
    pub fn entry<'a, K>(&'a mut self, key: K) -> Entry<'a, T, Extractor, K, S>
    where
        K: core::hash::Hash,
        for<'z> <Extractor as KeyExtractor<'z, T>>::Key: core::hash::Hash + PartialEq<K>,
    {
        <Self as IEntry<T, Extractor, S, DefaultBorrower>>::entry(self, key)
    }
    /// Similar to [`KeyedSet::insert`], but returns a mutable reference to the inserted value instead of the previous value.
    pub fn write(&mut self, value: T) -> &mut T
    where
        for<'a, 'b> <Extractor as KeyExtractor<'a, T>>::Key:
            PartialEq<<Extractor as KeyExtractor<'b, T>>::Key>,
    {
        let key = self.extractor.extract(&value);
        let mut hasher = self.hash_builder.build_hasher();
        key.hash(&mut hasher);
        let hash = hasher.finish();
        match self
            .inner
            .get_mut(hash, |i| self.extractor.extract(i).eq(&key))
        {
            Some(bucket) => {
                core::mem::drop(key);
                *bucket = value;
                unsafe { core::mem::transmute(bucket) }
            }
            None => {
                core::mem::drop(key);
                let hasher = make_hasher(&self.hash_builder, &self.extractor);
                let bucket = self.inner.insert(hash, value, hasher);
                unsafe { &mut *bucket.as_ptr() }
            }
        }
    }
    /// Access the value associated to the key immutably.
    pub fn get<K>(&self, key: &K) -> Option<&T>
    where
        K: core::hash::Hash,
        for<'a> <Extractor as KeyExtractor<'a, T>>::Key: core::hash::Hash + PartialEq<K>,
    {
        let mut hasher = self.hash_builder.build_hasher();
        key.hash(&mut hasher);
        let hash = hasher.finish();
        self.inner.get(hash, |i| self.extractor.extract(i).eq(key))
    }
    /// Access the value associated to the key mutably.
    ///
    /// The returned [`KeyedSetGuard`] will panic on drop if the value is modified in a way that modifies its key.
    pub fn get_mut<'a, K>(&'a mut self, key: &'a K) -> Option<KeyedSetGuard<'a, K, T, Extractor>>
    where
        K: core::hash::Hash,
        for<'z> <Extractor as KeyExtractor<'z, T>>::Key: core::hash::Hash + PartialEq<K>,
    {
        let mut hasher = self.hash_builder.build_hasher();
        key.hash(&mut hasher);
        let hash = hasher.finish();
        self.inner
            .get_mut(hash, |i| self.extractor.extract(i).eq(key))
            .map(|guarded| KeyedSetGuard {
                guarded,
                key,
                extractor: &self.extractor,
            })
    }
    /// Access the value associated to the key mutably.
    ///
    /// # Safety
    /// Mutating the value in a way that mutates its key may lead to undefined behaviour.
    pub unsafe fn get_mut_unguarded<'a, K>(&'a mut self, key: &K) -> Option<&'a mut T>
    where
        K: core::hash::Hash,
        for<'z> <Extractor as KeyExtractor<'z, T>>::Key: core::hash::Hash + PartialEq<K>,
    {
        let mut hasher = self.hash_builder.build_hasher();
        key.hash(&mut hasher);
        let hash = hasher.finish();
        self.inner
            .get_mut(hash, |i| self.extractor.extract(i).eq(key))
    }
    /// Remove the value associated to the key, returning it if it exists.
    pub fn remove<K>(&mut self, key: &K) -> Option<T>
    where
        K: core::hash::Hash,
        for<'z> <Extractor as KeyExtractor<'z, T>>::Key: core::hash::Hash + PartialEq<K>,
    {
        let mut hasher = self.hash_builder.build_hasher();
        key.hash(&mut hasher);
        let hash = hasher.finish();
        self.inner
            .remove_entry(hash, |i| self.extractor.extract(i).eq(key))
    }
    /// Returns an iterator that drains elements that match the provided predicate, while removing them from the set.
    ///
    /// Note that [`DrainFilter`] WILL iterate fully on drop, ensuring that all elements matching your predicate are always removed, even if you fail to iterate.
    pub fn drain_where<F: FnMut(&mut T) -> bool>(&mut self, predicate: F) -> DrainFilter<T, F> {
        DrainFilter {
            predicate,
            iter: unsafe { self.inner.iter() },
            table: &mut self.inner,
        }
    }
    /// Returns an iterator that drains elements from the collection, without affecting the collection's capacity.
    ///
    /// Note that [`Drain`] WILL iterate fully on drop, ensuring that all elements are indeed removed, even if you fail to iterate.
    pub fn drain(&mut self) -> Drain<T> {
        Drain {
            iter: unsafe { self.inner.iter() },
            table: &mut self.inner,
        }
    }
}
/// An iterator over a [`KeyedSet`] that steals the values from it.
pub struct Drain<'a, T> {
    iter: RawIter<T>,
    table: &'a mut RawTable<T>,
}

impl<'a, T> Drop for Drain<'a, T> {
    fn drop(&mut self) {
        for _ in self {}
    }
}

impl<'a, T> Iterator for Drain<'a, T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        Some(unsafe { self.table.remove(self.iter.next()?).0 })
    }
}
/// An iterator over a [`KeyedSet`] that only steals values that match a given predicate.
pub struct DrainFilter<'a, T, F: FnMut(&mut T) -> bool> {
    predicate: F,
    iter: RawIter<T>,
    table: &'a mut RawTable<T>,
}

impl<'a, T, F: FnMut(&mut T) -> bool> Drop for DrainFilter<'a, T, F> {
    fn drop(&mut self) {
        for _ in self {}
    }
}

impl<'a, T, F: FnMut(&mut T) -> bool> Iterator for DrainFilter<'a, T, F> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            for item in &mut self.iter {
                if (self.predicate)(item.as_mut()) {
                    return Some(self.table.remove(item).0);
                }
            }
        }
        None
    }
}
/// The trait magic that allows [`KeyedSet::entry`] to work.
pub trait IEntry<T, Extractor, S, Borrower = DefaultBorrower>
where
    Extractor: for<'a> KeyExtractor<'a, T>,
    for<'a> <Extractor as KeyExtractor<'a, T>>::Key: core::hash::Hash,
    S: BuildHasher,
{
    /// Access the entry for `key`.
    fn entry<'a, K>(&'a mut self, key: K) -> Entry<'a, T, Extractor, K, S>
    where
        Borrower: IBorrower<K>,
        <Borrower as IBorrower<K>>::Borrowed: core::hash::Hash,
        for<'z> <Extractor as KeyExtractor<'z, T>>::Key:
            core::hash::Hash + PartialEq<<Borrower as IBorrower<K>>::Borrowed>;
}
impl<T, Extractor, S, Borrower> IEntry<T, Extractor, S, Borrower> for KeyedSet<T, Extractor, S>
where
    Extractor: for<'a> KeyExtractor<'a, T>,
    for<'a> <Extractor as KeyExtractor<'a, T>>::Key: core::hash::Hash,
    S: BuildHasher,
{
    fn entry<'a, K>(&'a mut self, key: K) -> Entry<'a, T, Extractor, K, S>
    where
        Borrower: IBorrower<K>,
        <Borrower as IBorrower<K>>::Borrowed: core::hash::Hash,
        for<'z> <Extractor as KeyExtractor<'z, T>>::Key:
            core::hash::Hash + PartialEq<<Borrower as IBorrower<K>>::Borrowed>,
    {
        match unsafe { self.get_mut_unguarded(Borrower::borrow(&key)) } {
            Some(entry) => Entry::OccupiedEntry(unsafe { core::mem::transmute(entry) }),
            None => Entry::Vacant(VacantEntry { set: self, key }),
        }
    }
}
/// The default way to borrow a value.
pub struct DefaultBorrower;
/// Allows defining alternatives to [`core::ops::Deref`]
pub trait IBorrower<T> {
    /// The borrow target.
    type Borrowed;
    /// Borrows a value in its borrowed representation.
    fn borrow(value: &T) -> &Self::Borrowed;
}
impl<T> IBorrower<T> for DefaultBorrower {
    type Borrowed = T;

    fn borrow(value: &T) -> &Self::Borrowed {
        value
    }
}
impl<T, Extractor, S> KeyedSet<T, Extractor, S> {
    /// Iterate over the [`KeyedSet`]'s values immutably.
    pub fn iter(&self) -> Iter<T> {
        Iter {
            inner: unsafe { self.inner.iter() },
            marker: PhantomData,
        }
    }
    /// Iterate over the [`KeyedSet`]'s values mutably.
    pub fn iter_mut(&mut self) -> IterMut<T> {
        IterMut {
            inner: unsafe { self.inner.iter() },
            marker: PhantomData,
        }
    }
    /// Returns the number of elements in the [`KeyedSet`]
    pub fn len(&self) -> usize {
        self.inner.len()
    }
    /// Returns `true` if the [`KeyedSet`] is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}
/// A guard that allows mutating a value, but which panics if the new value once dropped doesn't have the same key.
pub struct KeyedSetGuard<'a, K, T, Extractor>
where
    Extractor: for<'z> KeyExtractor<'z, T>,
    for<'z> <Extractor as KeyExtractor<'z, T>>::Key: core::hash::Hash + PartialEq<K>,
{
    guarded: &'a mut T,
    key: &'a K,
    extractor: &'a Extractor,
}
impl<'a, K, T, Extractor> core::ops::Deref for KeyedSetGuard<'a, K, T, Extractor>
where
    Extractor: for<'z> KeyExtractor<'z, T>,
    for<'z> <Extractor as KeyExtractor<'z, T>>::Key: core::hash::Hash + PartialEq<K>,
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.guarded
    }
}
impl<'a, K, T, Extractor> core::ops::DerefMut for KeyedSetGuard<'a, K, T, Extractor>
where
    Extractor: for<'z> KeyExtractor<'z, T>,
    for<'z> <Extractor as KeyExtractor<'z, T>>::Key: core::hash::Hash + PartialEq<K>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.guarded
    }
}
impl<'a, K, T, Extractor> Drop for KeyedSetGuard<'a, K, T, Extractor>
where
    Extractor: for<'z> KeyExtractor<'z, T>,
    for<'z> <Extractor as KeyExtractor<'z, T>>::Key: core::hash::Hash + PartialEq<K>,
{
    fn drop(&mut self) {
        if !self.extractor.extract(&*self.guarded).eq(self.key) {
            panic!("KeyedSetGuard dropped with new value that would change the key, breaking the internal table's invariants.")
        }
    }
}

/// An iterator over the [`KeyedSet`] by value.
pub struct IntoIter<T>(RawIntoIter<T>);

impl<T> ExactSizeIterator for IntoIter<T> {
    fn len(&self) -> usize {
        self.0.len()
    }
}
impl<T> Iterator for IntoIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

/// An iterator over the [`KeyedSet`] by reference.
pub struct Iter<'a, T> {
    inner: RawIter<T>,
    marker: PhantomData<&'a ()>,
}
impl<'a, T: 'a> Iterator for Iter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|b| unsafe { b.as_ref() })
    }
}
impl<'a, T: 'a> ExactSizeIterator for Iter<'a, T> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}
/// An iterator over the [`KeyedSet`] by mutable reference.
pub struct IterMut<'a, T> {
    inner: RawIter<T>,
    marker: PhantomData<&'a mut ()>,
}
impl<'a, T: 'a> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|b| unsafe { b.as_mut() })
    }
}
impl<'a, T: 'a> ExactSizeIterator for IterMut<'a, T> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

/// A vacant entry into a [`KeyedSet`]
pub struct VacantEntry<'a, T: 'a, Extractor, K, S> {
    /// The inner set
    pub set: &'a mut KeyedSet<T, Extractor, S>,
    /// The key fort he entry.
    pub key: K,
}
/// An entry into a [`KeyedSet`], allowing in-place modification of the value associated with the key if it exists.
pub enum Entry<'a, T, Extractor, K, S = DefaultHashBuilder> {
    /// The key was not yet present in the [`KeyedSet`].
    Vacant(VacantEntry<'a, T, Extractor, K, S>),
    /// The key was already present in the [`KeyedSet`].
    OccupiedEntry(&'a mut T),
}

impl<'a, T: 'a, Extractor, S, K> Entry<'a, T, Extractor, K, S>
where
    S: BuildHasher,
    for<'z> Extractor: KeyExtractor<'z, T>,
    for<'z, 'b> <Extractor as KeyExtractor<'z, T>>::Key:
        PartialEq<<Extractor as KeyExtractor<'b, T>>::Key>,
{
    /// Get a mutable reference to the value if present, or assign a value constructed by `f` if it wasn't.
    pub fn get_or_insert_with(self, f: impl FnOnce(K) -> T) -> &'a mut T {
        match self {
            Entry::Vacant(entry) => entry.insert_with(f),
            Entry::OccupiedEntry(entry) => entry,
        }
    }
    /// A shortcut for `entry.get_or_insert_with(Into::into)`
    pub fn get_or_insert_with_into(self) -> &'a mut T
    where
        K: Into<T>,
    {
        self.get_or_insert_with(|k| k.into())
    }
}
impl<'a, K, T, Extractor, S> VacantEntry<'a, T, Extractor, K, S>
where
    S: BuildHasher,
    for<'z> Extractor: KeyExtractor<'z, T>,
    for<'z, 'b> <Extractor as KeyExtractor<'z, T>>::Key:
        PartialEq<<Extractor as KeyExtractor<'b, T>>::Key>,
{
    /// Inserts a value constructed from the entry's key using `f`.
    pub fn insert_with<F: FnOnce(K) -> T>(self, f: F) -> &'a mut T {
        self.set.write(f(self.key))
    }
}

#[allow(clippy::manual_hash_one)]
fn make_hasher<'a, S: BuildHasher, Extractor, T>(
    hash_builder: &'a S,
    extractor: &'a Extractor,
) -> impl Fn(&T) -> u64 + 'a
where
    Extractor: for<'b> KeyExtractor<'b, T>,
    for<'b> <Extractor as KeyExtractor<'b, T>>::Key: core::hash::Hash,
{
    move |value| {
        let key = extractor.extract(value);
        let mut hasher = hash_builder.build_hasher();
        key.hash(&mut hasher);
        hasher.finish()
    }
}

#[test]
fn test() {
    let mut set = KeyedSet::new(|value: &(u64, u64)| value.0);
    assert_eq!(set.len(), 0);
    set.insert((0, 0));
    assert_eq!(set.insert((0, 1)), Some((0, 0)));
    assert_eq!(set.len(), 1);
    assert_eq!(set.get(&0), Some(&(0, 1)));
    assert!(set.get(&1).is_none());
    assert_eq!(*set.entry(12).get_or_insert_with(|k| (k, k)), (12, 12));
}
