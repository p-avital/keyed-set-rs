use core::{
    hash::{BuildHasher, Hash, Hasher},
    marker::PhantomData,
};

use hashbrown::{
    hash_map::DefaultHashBuilder,
    raw::{RawIntoIter, RawIter},
};

#[derive(Clone, Default)]
pub struct KeyedSet<T, Extractor, S = DefaultHashBuilder> {
    inner: hashbrown::raw::RawTable<T>,
    hash_builder: S,
    extractor: Extractor,
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
pub trait KeyExtractor<'a, T> {
    type Key: Hash;
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
    for<'a> <Extractor as KeyExtractor<'a, T>>::Key: std::hash::Hash,
{
    pub fn new(extractor: Extractor) -> Self {
        Self {
            inner: Default::default(),
            hash_builder: Default::default(),
            extractor,
        }
    }
}

impl<T: std::fmt::Debug, Extractor, S> std::fmt::Debug for KeyedSet<T, Extractor, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "KeyedSet {{")?;
        for v in self.iter() {
            write!(f, "{:?}, ", v)?;
        }
        write!(f, "}}")
    }
}

impl<T, Extractor, S> KeyedSet<T, Extractor, S>
where
    Extractor: for<'a> KeyExtractor<'a, T>,
    for<'a> <Extractor as KeyExtractor<'a, T>>::Key: std::hash::Hash,
    S: BuildHasher,
{
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
    pub fn entry<'a, K>(&'a mut self, key: K) -> Entry<'a, T, Extractor, K, S>
    where
        K: std::hash::Hash,
        for<'z> <Extractor as KeyExtractor<'z, T>>::Key: std::hash::Hash + PartialEq<K>,
    {
        <Self as IEntry<T, Extractor, S, DefaultBorrower>>::entry(self, key)
    }
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
                unsafe { std::mem::transmute(bucket) }
            }
            None => {
                core::mem::drop(key);
                let hasher = make_hasher(&self.hash_builder, &self.extractor);
                let bucket = self.inner.insert(hash, value, hasher);
                unsafe { &mut *bucket.as_ptr() }
            }
        }
    }
    pub fn get<K>(&self, key: &K) -> Option<&T>
    where
        K: std::hash::Hash,
        for<'a> <Extractor as KeyExtractor<'a, T>>::Key: std::hash::Hash + PartialEq<K>,
    {
        let mut hasher = self.hash_builder.build_hasher();
        key.hash(&mut hasher);
        let hash = hasher.finish();
        self.inner.get(hash, |i| self.extractor.extract(i).eq(key))
    }
    pub fn get_mut<'a, K>(&'a mut self, key: &'a K) -> Option<KeyedSetGuard<'a, K, T, Extractor>>
    where
        K: std::hash::Hash,
        for<'z> <Extractor as KeyExtractor<'z, T>>::Key: std::hash::Hash + PartialEq<K>,
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
    pub fn get_mut_unguarded<'a, K>(&'a mut self, key: &K) -> Option<&'a mut T>
    where
        K: std::hash::Hash,
        for<'z> <Extractor as KeyExtractor<'z, T>>::Key: std::hash::Hash + PartialEq<K>,
    {
        let mut hasher = self.hash_builder.build_hasher();
        key.hash(&mut hasher);
        let hash = hasher.finish();
        self.inner
            .get_mut(hash, |i| self.extractor.extract(i).eq(key))
    }
}
pub trait IEntry<T, Extractor, S, Borrower = DefaultBorrower>
where
    Extractor: for<'a> KeyExtractor<'a, T>,
    for<'a> <Extractor as KeyExtractor<'a, T>>::Key: std::hash::Hash,
    S: BuildHasher,
{
    fn entry<'a, K>(&'a mut self, key: K) -> Entry<'a, T, Extractor, K, S>
    where
        Borrower: IBorrower<K>,
        <Borrower as IBorrower<K>>::Borrowed: std::hash::Hash,
        for<'z> <Extractor as KeyExtractor<'z, T>>::Key:
            std::hash::Hash + PartialEq<<Borrower as IBorrower<K>>::Borrowed>;
}
impl<T, Extractor, S, Borrower> IEntry<T, Extractor, S, Borrower> for KeyedSet<T, Extractor, S>
where
    Extractor: for<'a> KeyExtractor<'a, T>,
    for<'a> <Extractor as KeyExtractor<'a, T>>::Key: std::hash::Hash,
    S: BuildHasher,
{
    fn entry<'a, K>(&'a mut self, key: K) -> Entry<'a, T, Extractor, K, S>
    where
        Borrower: IBorrower<K>,
        <Borrower as IBorrower<K>>::Borrowed: std::hash::Hash,
        for<'z> <Extractor as KeyExtractor<'z, T>>::Key:
            std::hash::Hash + PartialEq<<Borrower as IBorrower<K>>::Borrowed>,
    {
        match self.get_mut_unguarded(Borrower::borrow(&key)) {
            Some(entry) => Entry::OccupiedEntry(unsafe { std::mem::transmute(entry) }),
            None => Entry::Vacant(VacantEntry { set: self, key }),
        }
    }
}
pub struct DefaultBorrower;
pub trait IBorrower<T> {
    type Borrowed;
    fn borrow(value: &T) -> &Self::Borrowed;
}
impl<T> IBorrower<T> for DefaultBorrower {
    type Borrowed = T;

    fn borrow(value: &T) -> &Self::Borrowed {
        value
    }
}
impl<T, Extractor, S> KeyedSet<T, Extractor, S> {
    pub fn iter(&self) -> Iter<T> {
        Iter {
            inner: unsafe { self.inner.iter() },
            marker: PhantomData,
        }
    }
    pub fn iter_mut(&mut self) -> IterMut<T> {
        IterMut {
            inner: unsafe { self.inner.iter() },
            marker: PhantomData,
        }
    }
    pub fn len(&self) -> usize {
        self.inner.len()
    }
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

pub struct KeyedSetGuard<'a, K, T, Extractor>
where
    Extractor: for<'z> KeyExtractor<'z, T>,
    for<'z> <Extractor as KeyExtractor<'z, T>>::Key: std::hash::Hash + PartialEq<K>,
{
    guarded: &'a mut T,
    key: &'a K,
    extractor: &'a Extractor,
}
impl<'a, K, T, Extractor> std::ops::Deref for KeyedSetGuard<'a, K, T, Extractor>
where
    Extractor: for<'z> KeyExtractor<'z, T>,
    for<'z> <Extractor as KeyExtractor<'z, T>>::Key: std::hash::Hash + PartialEq<K>,
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.guarded
    }
}
impl<'a, K, T, Extractor> std::ops::DerefMut for KeyedSetGuard<'a, K, T, Extractor>
where
    Extractor: for<'z> KeyExtractor<'z, T>,
    for<'z> <Extractor as KeyExtractor<'z, T>>::Key: std::hash::Hash + PartialEq<K>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.guarded
    }
}
impl<'a, K, T, Extractor> Drop for KeyedSetGuard<'a, K, T, Extractor>
where
    Extractor: for<'z> KeyExtractor<'z, T>,
    for<'z> <Extractor as KeyExtractor<'z, T>>::Key: std::hash::Hash + PartialEq<K>,
{
    fn drop(&mut self) {
        if !self.extractor.extract(&*self.guarded).eq(self.key) {
            panic!("KeyedSetGuard dropped with new value that would change the key, breaking the internal table's invariants.")
        }
    }
}

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

pub struct VacantEntry<'a, T: 'a, Extractor, K, S> {
    pub set: &'a mut KeyedSet<T, Extractor, S>,
    pub key: K,
}
pub enum Entry<'a, T, Extractor, K, S = DefaultHashBuilder> {
    Vacant(VacantEntry<'a, T, Extractor, K, S>),
    OccupiedEntry(&'a mut T),
}

impl<'a, T: 'a, Extractor, S, K> Entry<'a, T, Extractor, K, S>
where
    S: BuildHasher,
    for<'z> Extractor: KeyExtractor<'z, T>,
    for<'z, 'b> <Extractor as KeyExtractor<'z, T>>::Key:
        PartialEq<<Extractor as KeyExtractor<'b, T>>::Key>,
{
    pub fn get_or_insert_with(self, f: impl FnOnce(K) -> T) -> &'a mut T {
        match self {
            Entry::Vacant(entry) => entry.insert_with(f),
            Entry::OccupiedEntry(entry) => entry,
        }
    }
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
    pub fn insert_with<F: FnOnce(K) -> T>(self, f: F) -> &'a mut T {
        self.set.write(f(self.key))
    }
}

fn make_hasher<'a, S: BuildHasher, Extractor, T>(
    hash_builder: &'a S,
    extractor: &'a Extractor,
) -> impl Fn(&T) -> u64 + 'a
where
    Extractor: for<'b> KeyExtractor<'b, T>,
    for<'b> <Extractor as KeyExtractor<'b, T>>::Key: std::hash::Hash,
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
    dbg!(&set);
}
