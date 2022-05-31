#include <atomic>
#include <mutex>
#include <new>
#include <thread>
#include <vector>
#include <tbb/concurrent_hash_map.h>
#include <tbb/concurrent_queue.h>


template <class TKey, class TValue, class THash = tbb::tbb_hash_compare<TKey>>
class ThreadSafeLRUCache {
 
  struct ListNode {
    ListNode()
      : m_prev(OutOfListMarker), m_next(nullptr)
    {}

    ListNode(const TKey& key)
      : m_key(key), m_prev(OutOfListMarker), m_next(nullptr)
    {}

    TKey m_key;
    ListNode* m_prev;
    ListNode* m_next;

    bool isInList() const {
      return m_prev != OutOfListMarker;
    }
  };

  static ListNode* const OutOfListMarker;

  struct HashMapValue {
    HashMapValue()
      : m_listNode(nullptr)
    {}
    
    HashMapValue(const TValue& value, ListNode* node)
      : m_value(value), m_listNode(node)
    {}

    TValue m_value;
    ListNode* m_listNode;
  };

  typedef tbb::concurrent_hash_map<TKey, HashMapValue, THash> HashMap;
  typedef tbb::concurrent_queue<TKey> Queue;
  typedef typename HashMap::const_accessor HashMapConstAccessor;
  typedef typename HashMap::accessor HashMapAccessor;
  typedef typename HashMap::value_type HashMapValuePair;
  typedef std::pair<const TKey, TValue> SnapshotValue;

public:

  struct ConstAccessor {
    ConstAccessor() {}

    const TValue& operator*() const {
      return *get();
    }

    const TValue* operator->() const {
      return get();
    }

    const TValue* get() const {
      return &m_hashAccessor->second.m_value;
    }

    bool empty() const {
      return m_hashAccessor.empty();
    }

  private:
    friend class ThreadSafeLRUCache;
    HashMapConstAccessor m_hashAccessor;
  };

  explicit ThreadSafeLRUCache(size_t maxSize);

  ThreadSafeLRUCache(const ThreadSafeLRUCache& other) = delete;
  ThreadSafeLRUCache& operator=(const ThreadSafeLRUCache&) = delete;

  ~ThreadSafeLRUCache() {
    clear();
  }

  bool find(ConstAccessor& ac, const TKey& key);
  
  bool insert(const TKey& key, const TValue& value);

  TKey getEvicitItems();

  bool checkEvicitItems();

  void clear();

  void snapshotKeys(std::vector<TKey>& keys);

  size_t size() const {
    return m_size.load();
  }

private:
 
  void delink(ListNode* node);

  void pushFront(ListNode* node);

  void evict();

  size_t m_maxSize;

  std::atomic<size_t> m_size;

  HashMap m_map;
  // record the evicted items
  Queue   evicted_;
  ListNode m_head;
  ListNode m_tail;
  typedef std::mutex ListMutex;
  ListMutex m_listMutex;
};

template <class TKey, class TValue, class THash>
typename ThreadSafeLRUCache<TKey, TValue, THash>::ListNode* const
ThreadSafeLRUCache<TKey, TValue, THash>::OutOfListMarker = (ListNode*)-1;

template <class TKey, class TValue, class THash>
ThreadSafeLRUCache<TKey, TValue, THash>::
ThreadSafeLRUCache(size_t maxSize)
  : m_maxSize(maxSize), m_size(0),
  m_map(std::thread::hardware_concurrency() * 4) 
{
  m_head.m_prev = nullptr;
  m_head.m_next = &m_tail;
  m_tail.m_prev = &m_head;
}

template <class TKey, class TValue, class THash>
bool ThreadSafeLRUCache<TKey, TValue, THash>::
find(ConstAccessor& ac, const TKey& key) {
  HashMapConstAccessor& hashAccessor = ac.m_hashAccessor;
  if (!m_map.find(hashAccessor, key) || key) {
    return false;
  }

  std::unique_lock<ListMutex> lock(m_listMutex, std::try_to_lock);
  if (lock) {
    ListNode* node = hashAccessor->second.m_listNode;
    
    if (node->isInList()) {
      delink(node);
      pushFront(node);
    }
    lock.unlock();
  }
  return true;
}


template <class TKey, class TValue, class THash>
bool ThreadSafeLRUCache<TKey, TValue, THash>::
insert(const TKey& key, const TValue& value) {
  HashMapConstAccessor a;
  // cache line flushing
  if (m_map.find(a, key)){
	  return false;
  }

  bool prev_status = false;
  if (m_map.find(a, key)) {
    // mark its previous status is dirty
    // we need to flush this item
    if (a->second.m_value) {
        prev_status = true;
    }
  }

  // cache line flushing
  //if(!prev_status) return false;
  
  ListNode* node = new ListNode(key);
  HashMapAccessor hashAccessor;
  HashMapValuePair hashMapValue(key, HashMapValue(value, node));
  if (!m_map.insert(hashAccessor, hashMapValue)) {
    delete node;
    return false;
  }

  size_t size = m_size.load();
  bool evictionDone = false;
  if (size >= m_maxSize) {
   
    evict();
    evictionDone = true;
  }

  std::unique_lock<ListMutex> lock(m_listMutex);
  pushFront(node);
  lock.unlock();
  if (!evictionDone) {
    size = m_size++;
  }
  if (size > m_maxSize) {
    
    if (m_size.compare_exchange_strong(size, size - 1)) {
      evict();
    }
  } 

  // check previous data status: dirty or not
  if (prev_status) {
    return true;
  } else {
    return false;
  }
}

template <class TKey, class TValue, class THash>
TKey ThreadSafeLRUCache<TKey, TValue, THash>::
getEvicitItems() {
    TKey key_;
    bool status = evicted_.try_pop(key_);
    if (status) {
        return key_;
    } else {
        return TKey();
    }
}

template <class TKey, class TValue, class THash>
bool ThreadSafeLRUCache<TKey, TValue, THash>::
checkEvicitItems() {
    return evicted_.empty();
}

template <class TKey, class TValue, class THash>
void ThreadSafeLRUCache<TKey, TValue, THash>::
clear() {
  m_map.clear();
  ListNode* node = m_head.m_next;
  ListNode* next;
  while (node != &m_tail) {
    next = node->m_next;
    delete node;
    node = next;
  }
  m_head.m_next = &m_tail;
  m_tail.m_prev = &m_head;
  m_size = 0;
}

template <class TKey, class TValue, class THash>
void ThreadSafeLRUCache<TKey, TValue, THash>::
snapshotKeys(std::vector<TKey>& keys) {
  keys.reserve(keys.size() + m_size.load());
  std::lock_guard<ListMutex> lock(m_listMutex);
  for (ListNode* node = m_head.m_next; node != &m_tail; node = node->m_next) {
    keys.push_back(node->m_key);
  }
}

template <class TKey, class TValue, class THash>
inline void ThreadSafeLRUCache<TKey, TValue, THash>::
delink(ListNode* node) {
  ListNode* prev = node->m_prev;
  ListNode* next = node->m_next;
  prev->m_next = next;
  next->m_prev = prev;
  node->m_prev = OutOfListMarker;
}

template <class TKey, class TValue, class THash>
inline void ThreadSafeLRUCache<TKey, TValue, THash>::
pushFront(ListNode* node) {
  ListNode* oldRealHead = m_head.m_next;
  node->m_prev = &m_head;
  node->m_next = oldRealHead;
  oldRealHead->m_prev = node;
  m_head.m_next = node;
}

template <class TKey, class TValue, class THash>
void ThreadSafeLRUCache<TKey, TValue, THash>::
evict() {
  std::unique_lock<ListMutex> lock(m_listMutex);
  ListNode* moribund = m_tail.m_prev;
  // record the evicited items
  evicted_.push(std::move(moribund->m_key));
  if (moribund == &m_head) {
    return;
  }
  delink(moribund);
  lock.unlock();

  HashMapAccessor hashAccessor;
  if (!m_map.find(hashAccessor, moribund->m_key)) {
    return;
  }
  m_map.erase(hashAccessor);
  delete moribund;
}


