use fnv::FnvHashMap;
use parking_lot::RwLock;
use std::sync::Arc;

/// Thread-safe string interner for reducing allocations
/// of frequently-used strings
#[derive(Clone)]
pub struct StringInterner {
    strings: Arc<RwLock<FnvHashMap<String, Arc<str>>>>,
}

impl StringInterner {
    pub fn new() -> Self {
        Self {
            strings: Arc::new(RwLock::new(FnvHashMap::default())),
        }
    }

    /// Intern a string, returning a cheap-to-clone Arc
    pub fn intern(&self, s: &str) -> Arc<str> {
        {
            let read = self.strings.read();
            if let Some(interned) = read.get(s) {
                return Arc::clone(interned);
            }
        }

        let mut write = self.strings.write();
        write
            .entry(s.to_string())
            .or_insert_with(|| Arc::from(s))
            .clone()
    }

    /// Get stats for monitoring
    pub fn stats(&self) -> InternerStats {
        let read = self.strings.read();
        InternerStats {
            unique_strings: read.len(),
            total_bytes: read.keys().map(|k| k.len()).sum(),
        }
    }
}

impl Default for StringInterner {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct InternerStats {
    pub unique_strings: usize,
    pub total_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intern_same_string() {
        let interner = StringInterner::new();
        let s1 = interner.intern("hello");
        let s2 = interner.intern("hello");
        assert!(Arc::ptr_eq(&s1, &s2));
    }

    #[test]
    fn test_intern_different_strings() {
        let interner = StringInterner::new();
        let s1 = interner.intern("hello");
        let s2 = interner.intern("world");
        assert!(!Arc::ptr_eq(&s1, &s2));
        assert_eq!(&*s1, "hello");
        assert_eq!(&*s2, "world");
    }

    #[test]
    fn test_stats() {
        let interner = StringInterner::new();
        interner.intern("hello");
        interner.intern("world");
        interner.intern("hello");
        let stats = interner.stats();
        assert_eq!(stats.unique_strings, 2);
        assert_eq!(stats.total_bytes, 10);
    }
}
