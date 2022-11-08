use core::{
    hash::{BuildHasher, Hash, Hasher},
    marker::PhantomData,
};

use hashbrown::{hash_map::DefaultHashBuilder, raw::RawIter};

#[derive(Clone, Default)]
pub struct KeyedSet<T, Extractor, S = DefaultHashBuilder> {
    inner: hashbrown::raw::RawTable<T>,
    hash_builder: S,
    extractor: Extractor,
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
    dbg!(&set);
    assert_eq!(set.len(), 1);
    assert_eq!(set.get(&0), Some(&(0, 1)));
    assert!(set.get(&1).is_none());
}
