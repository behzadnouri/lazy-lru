#![no_std]
use {
    alloc::vec::Vec,
    core::{
        borrow::Borrow,
        cmp::Reverse,
        hash::Hash,
        sync::atomic::{AtomicU64, Ordering},
    },
    hashbrown::HashMap,
};

extern crate alloc;

/// A least-recently-used (LRU) Cache with lazy eviction.
/// Each entry maintains an associated ordinal value representing when the
/// entry was last accessed. The cache is allowed to grow up to 2 times
/// specified capacity with no evictions, at which point, the excess entries
/// are evicted based on LRU policy resulting in an _amortized_ `O(1)`
/// performance.
pub struct LruCache<K, V> {
    cache: HashMap<K, (/*ordinal:*/ AtomicU64, V)>,
    counter: AtomicU64,
    capacity: usize,
}

impl<K, V> LruCache<K, V> {
    pub fn new(capacity: usize) -> LruCache<K, V> {
        Self {
            cache: HashMap::with_capacity(capacity.saturating_mul(2)),
            counter: AtomicU64::default(),
            capacity,
        }
    }
}

impl<K: Eq + Hash + PartialEq, V> LruCache<K, V> {
    // Inserts a key-value pair into the cache.
    // If the cache not have this key present, None is returned.
    // If the cache did have this key present, the value is updated, and the
    // old value is returned. The key is not updated, though; this matters for
    // types that can be == without being identical.
    pub fn put(&mut self, key: K, value: V) -> Option<V> {
        let ordinal = self.counter.fetch_add(1, Ordering::Relaxed);
        let old = self
            .cache
            .insert(key, (AtomicU64::new(ordinal), value))
            .map(|(_, value)| value);
        self.maybe_evict();
        old
    }

    // If the cache hash grown to at least twice the self.capacity, evicts
    // extra entries from the cache by LRU policy.
    fn maybe_evict(&mut self) {
        if self.cache.len() < self.capacity.saturating_mul(2) {
            return;
        }
        let mut entries: Vec<(K, (/*ordinal:*/ u64, V))> = self
            .cache
            .drain()
            .map(|(key, (ordinal, value))| (key, (ordinal.into_inner(), value)))
            .collect();
        entries
            .select_nth_unstable_by_key(self.capacity.saturating_sub(1), |&(_, (ordinal, _))| {
                Reverse(ordinal)
            });
        self.cache.extend(
            entries
                .into_iter()
                .take(self.capacity)
                .map(|(key, (ordinal, value))| (key, (AtomicU64::new(ordinal), value))),
        );
    }

    /// Returns true if the map contains a value for the specified key.
    /// The key may be any borrowed form of the cache's key type, but `Hash`
    /// and `Eq` on the borrowed form must match those for the key type.
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.cache.contains_key(key)
    }

    /// Returns a reference to the value corresponding to the key.
    /// Updates the ordinal value associated with the entry.
    /// The key may be any borrowed form of the cache's key type, but `Hash`
    /// and `Eq` on the borrowed form must match those for the key type.
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let (ordinal, value) = self.cache.get(key)?;
        // fetch_max instead of store here because of possible concurrent
        // lookups of the same key.
        ordinal.fetch_max(
            self.counter.fetch_add(1, Ordering::Relaxed),
            Ordering::Relaxed,
        );
        Some(value)
    }

    /// Returns a mutable reference to the value corresponding to the key.
    /// Updates the ordinal value associated with the entry.
    /// The key may be any borrowed form of the cache's key type, but `Hash`
    /// and `Eq` on the borrowed form must match those for the key type.
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let (ordinal, value) = self.cache.get_mut(key)?;
        // store is sufficient here because &mut self prevents concurrent
        // update.
        ordinal.store(
            self.counter.fetch_add(1, Ordering::Relaxed),
            Ordering::Relaxed,
        );
        Some(value)
    }

    /// Returns the key-value pair corresponding to the supplied key.
    /// Updates the ordinal value associated with the entry.
    /// The supplied key may be any borrowed form of the cache's key type, but
    /// `Hash` and `Eq` on the borrowed form must match those for the key type.
    pub fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let (key, (ordinal, value)) = self.cache.get_key_value(key)?;
        // fetch_max instead of store here because of possible concurrent
        // lookups of the same key.
        ordinal.fetch_max(
            self.counter.fetch_add(1, Ordering::Relaxed),
            Ordering::Relaxed,
        );
        Some((key, value))
    }

    /// Returns a reference to the value corresponding to the key.
    /// Unlike `get`, `peek` does _not_ updates the ordinal value associated
    /// with the entry.
    pub fn peek<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.cache.get(key).map(|(_, value)| value)
    }

    /// Returns a mutable reference to the value corresponding to the key.
    /// Unlike `get_mut`, `peek_mut` does _not_ updates the ordinal value
    /// associated with the entry.
    /// The key may be any borrowed form of the cache's key type, but `Hash`
    /// and `Eq` on the borrowed form must match those for the key type.
    pub fn peek_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.cache.get_mut(key).map(|(_, value)| value)
    }

    /// Returns the key-value pair corresponding to the supplied key.
    /// Unlike `get_key_value`, `peek_key_value` does _not_ updates the ordinal
    /// value associated with the entry.
    /// The supplied key may be any borrowed form of the cache's key type, but
    /// `Hash` and `Eq` on the borrowed form must match those for the key type.
    pub fn peek_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.cache
            .get_key_value(key)
            .map(|(key, (_, value))| (key, value))
    }

    /// Removes a key from the cache, returning the value at the key if the key
    /// was previously in the cache.
    /// The key may be any borrowed form of the cache's key type, but `Hash`
    /// and `Eq` on the borrowed form must match those for the key type.
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.cache.remove(key).map(|(_, value)| value)
    }

    /// Removes a key from the cache, returning the stored key and value if the
    /// key was previously in the cache.
    /// The key may be any borrowed form of the cache's key type, but `Hash`
    /// and `Eq` on the borrowed form must match those for the key type.
    pub fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.cache
            .remove_entry(key)
            .map(|(key, (_, value))| (key, value))
    }

    /// Returns the number of elements in the cache.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Returns true if the cache contains no entries.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Clears the cache, removing all key-value pairs.
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Retains only the elements specified by the predicate.
    /// In other words, remove all pairs `(k, v)` for which `f(&k, &mut v)`
    /// returns `false`. The elements are visited in unsorted (and unspecified)
    /// order.
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        self.cache.retain(|key, (_, value)| f(key, value));
    }
}

#[cfg(test)]
mod tests {
    use {super::*, core::fmt::Debug};

    fn check_entry<K, V, Q: ?Sized>(cache: &LruCache<K, V>, key: &Q, ordinal: u64, value: V)
    where
        K: Hash + Eq + Borrow<Q>,
        Q: Hash + Eq,
        V: Debug + PartialEq<V>,
    {
        let (entry_ordinal, entry_value) = cache.cache.get(key).unwrap();
        assert_eq!(entry_value, &value);
        assert_eq!(entry_ordinal.load(Ordering::Relaxed), ordinal);
    }

    #[test]
    fn test_capacity_zero() {
        let mut cache = LruCache::new(0);

        cache.put("apple", 8);
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.get("apple"), None);
    }

    #[test]
    fn test_basics() {
        let mut cache = LruCache::new(2);

        cache.put("apple", 8);
        assert_eq!(cache.len(), 1);
        check_entry(&cache, "apple", 0, 8);
        assert_eq!(cache.counter.load(Ordering::Relaxed), 1);

        assert_eq!(cache.peek("apple"), Some(&8));
        check_entry(&cache, "apple", 0, 8);
        assert_eq!(cache.counter.load(Ordering::Relaxed), 1);

        assert_eq!(cache.peek_mut("apple"), Some(&mut 8));
        check_entry(&cache, "apple", 0, 8);
        assert_eq!(cache.counter.load(Ordering::Relaxed), 1);

        assert_eq!(cache.get("apple"), Some(&8));
        check_entry(&cache, "apple", 1, 8);
        assert_eq!(cache.counter.load(Ordering::Relaxed), 2);

        assert_eq!(cache.get_mut("apple"), Some(&mut 8));
        check_entry(&cache, "apple", 2, 8);
        assert_eq!(cache.counter.load(Ordering::Relaxed), 3);

        cache.put("banana", 4);
        assert_eq!(cache.len(), 2);
        check_entry(&cache, "banana", 3, 4);
        assert_eq!(cache.counter.load(Ordering::Relaxed), 4);

        cache.put("pear", 2);
        assert_eq!(cache.len(), 3);
        check_entry(&cache, "pear", 4, 2);
        assert_eq!(cache.counter.load(Ordering::Relaxed), 5);

        cache.put("banana", 6);
        assert_eq!(cache.len(), 3);
        check_entry(&cache, "banana", 5, 6);
        assert_eq!(cache.counter.load(Ordering::Relaxed), 6);

        cache.put("orange", 3); // triggers eviction
        assert_eq!(cache.len(), 2);
        check_entry(&cache, "banana", 5, 6);
        check_entry(&cache, "orange", 6, 3);
        assert_eq!(cache.counter.load(Ordering::Relaxed), 7);

        assert!(cache.contains_key("banana"));
        assert!(cache.contains_key("orange"));
        assert!(!cache.contains_key("apple"));
        assert!(!cache.contains_key("pear"));

        assert_eq!(cache.remove("banana"), Some(6));
        assert_eq!(cache.remove("banana"), None);

        assert_eq!(cache.remove_entry("orange"), Some(("orange", 3)));
        assert_eq!(cache.remove_entry("orange"), None);

        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }
}
