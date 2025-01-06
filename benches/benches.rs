#![feature(test)]

extern crate ahash;
extern crate core;
extern crate lazy_lru;
extern crate lru;
extern crate rand;
extern crate test;

use {core::num::NonZeroUsize, rand::Rng, test::Bencher};

const REPS: usize = 1 << 10;
const NUM_KEYS: usize = 1 << 20;
const CAPACITY: usize = 1 << 17;

macro_rules! impl_put_bench {
    ($name: ident, $cache: ty) => {
        fn $name(bencher: &mut Bencher, mut cache: $cache, num_keys: usize, reps: usize) {
            let mut rng = rand::thread_rng();
            cache.clear();
            for _ in 0..5 * CAPACITY {
                let key = rng.gen_range(0..num_keys);
                let _ = cache.put(key, ());
            }
            bencher.iter(|| {
                for _ in 0..reps {
                    let key = rng.gen_range(0..num_keys);
                    core::hint::black_box(cache.put(key, ()));
                }
            });
        }
    };
}

macro_rules! impl_get_bench {
    ($name: ident, $cache: ty) => {
        fn $name(bencher: &mut Bencher, mut cache: $cache, num_keys: usize, reps: usize) {
            let mut rng = rand::thread_rng();
            cache.clear();
            for _ in 0..5 * CAPACITY {
                let key = rng.gen_range(0..num_keys);
                let _ = cache.put(key, ());
            }
            bencher.iter(|| {
                for _ in 0..reps {
                    let key = rng.gen_range(0..num_keys);
                    core::hint::black_box(cache.get(&key));
                }
            });
        }
    };
}

impl_put_bench!(run_put_bench_eager, lru::LruCache<usize, ()>);
impl_put_bench!(run_put_bench_lazy, lazy_lru::LruCache<usize, ()>);
impl_put_bench!(run_put_bench_lazy_ahash, lazy_lru::LruCache<usize, (), ahash::RandomState>);

impl_get_bench!(run_get_bench_eager, lru::LruCache<usize, ()>);
impl_get_bench!(run_get_bench_lazy, lazy_lru::LruCache<usize, ()>);
impl_get_bench!(run_get_bench_lazy_ahash, lazy_lru::LruCache<usize, (), ahash::RandomState>);

#[bench]
fn bench_put_eager(bencher: &mut Bencher) {
    let cache = lru::LruCache::new(NonZeroUsize::new(CAPACITY).unwrap());
    run_put_bench_eager(bencher, cache, NUM_KEYS, REPS);
}

#[bench]
fn bench_put_lazy(bencher: &mut Bencher) {
    let cache = lazy_lru::LruCache::new(CAPACITY);
    run_put_bench_lazy(bencher, cache, NUM_KEYS, REPS);
}

#[bench]
fn bench_put_lazy_ahash(bencher: &mut Bencher) {
    let random_state = ahash::RandomState::new();
    let cache = lazy_lru::LruCache::with_capacity_and_hasher(CAPACITY, random_state);
    run_put_bench_lazy_ahash(bencher, cache, NUM_KEYS, REPS);
}

#[bench]
fn bench_get_eager(bencher: &mut Bencher) {
    let cache = lru::LruCache::new(NonZeroUsize::new(CAPACITY).unwrap());
    run_get_bench_eager(bencher, cache, NUM_KEYS, REPS);
}

#[bench]
fn bench_get_lazy(bencher: &mut Bencher) {
    let cache = lazy_lru::LruCache::new(CAPACITY);
    run_get_bench_lazy(bencher, cache, NUM_KEYS, REPS);
}

#[bench]
fn bench_get_lazy_ahash(bencher: &mut Bencher) {
    let random_state = ahash::RandomState::new();
    let cache = lazy_lru::LruCache::with_capacity_and_hasher(CAPACITY, random_state);
    run_get_bench_lazy_ahash(bencher, cache, NUM_KEYS, REPS);
}
