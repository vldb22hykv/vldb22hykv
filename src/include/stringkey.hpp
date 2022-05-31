#include <atomic>
#include <cstring>
#include <limits>
#include <memory>

#include "lru_hash.hpp"


struct ThreadSafeStringKey {
  ThreadSafeStringKey(const char* data, size_t size)
    : m_storage(new Storage(data, size))
  {}

ThreadSafeStringKey(const char* data)
    : m_storage(new Storage(data, 4))
  {}

  ThreadSafeStringKey() {}

  uint64_t hash() const {
    return m_storage->hash();
  }

  size_t size() const {
    return m_storage->m_size;
  }

  const char* data() const {
    return m_storage->m_data;
  }

  const char* c_str() const {
    return data();
  }

  bool operator==(const ThreadSafeStringKey& other) const {
    size_t s = size();
    return s == other.size() && 0 == std::memcmp(data(), other.data(), s);
  }

  struct HashCompare {
    bool equal(const ThreadSafeStringKey& j, const ThreadSafeStringKey& k) const {
      return j == k;
    }

    size_t hash(const ThreadSafeStringKey& k) const {
      return k.hash();
    }
  };

private:
  struct Storage {
    Storage(const char* data, size_t size) 
      : m_size(size), m_hash(0)
    {
      m_data = new char[size + 1];
      memcpy(m_data, data, size);
      m_data[size] = '\0';
    }

    ~Storage() {
      delete[] m_data;
    }

    char* m_data;
    size_t m_size;
    mutable std::atomic<size_t> m_hash;

    size_t hash() const {
      size_t h = m_hash.load(std::memory_order_relaxed);
      if (h == 0) {
        uint64_t h128[2];
        MurmurHash3::hash128<false>(m_data, m_size, 0, h128);
        h = (size_t)h128[0];
        if (h == 0) {
          h = 1;
        }
        m_hash.store(h, std::memory_order_relaxed);
      }
      return h;
    }
  };

  std::shared_ptr<Storage> m_storage;
};