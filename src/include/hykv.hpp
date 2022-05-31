#pragma once

#include <iostream>
#include <bitset>
#include <fcntl.h>
#include <unistd.h>
#include <vector>
#include <libpmem.h>
#include <thread>
#include <cmath>
#include <linux/mman.h>
#include <sys/mman.h>
#include <atomic>
#include <assert.h>
#include <filesystem>
#include <immintrin.h>

#include "cceh.hpp"
#include "concurrentqueue.h"

#include <numa.h>

#include "coru_infra.hpp"
#include "sclf.hpp"

#define ORIGINAL
//#define CACHE_LINE_FLUSHING
//#define COROUTINE
//#define DATA_MIGRATION


#ifndef NDEBUG
#define DEBUG_LOG(msg) (std::cout << msg << std::endl)
#else
#define DEBUG_LOG(msg) do {} while(0)
#endif

/**
 * Define this to use Hykv in DRAM instead of PMem. This will allocate all VPages in DRAM.
 */
#define HYKV_DRAM

#ifdef COROUTINE
inline throttler *throttlers;
#endif

namespace hykv {	

int hykv_numa = 0;
int default_numa_node_vpage = hykv_numa;
int numa_node_vpage = hykv_numa;	

using version_lock_t = uint8_t;

static constexpr uint16_t PAGE_SIZE = 4 * 1024; // 4kb
static constexpr uint8_t NUM_DIMMS = 6;
static constexpr size_t BLOCK_SIZE = NUM_DIMMS * PAGE_SIZE;
static constexpr size_t ONE_GB = 1024l * 1024 * 1024;

static_assert(sizeof(version_lock_t) == 1, "Lock must be 1 byte.");
static constexpr version_lock_t CLIENT_BIT    = 0b10000000;
static constexpr version_lock_t NO_CLIENT_BIT = 0b01111111;
static constexpr version_lock_t UNUSED_BIT    = 0b01000000;
static constexpr version_lock_t UNLOCKED_BIT  = 0b11111110;

#define IS_LOCKED(lock)       ((lock) & 1)

#define IO_ERROR(msg) throw std::runtime_error(std::string(msg) + " | " + std::strerror(errno) + " (LINE: " + std::to_string(__LINE__) + ')')
#define MMAP_CHECK(addr) if ((addr) == nullptr || (addr) == MAP_FAILED) { IO_ERROR("Cannot mmap"); }

#define LOAD_ORDER  std::memory_order_acquire
#define STORE_ORDER std::memory_order_release

static constexpr auto HYKV_MAP_PROT = PROT_WRITE | PROT_READ;
static constexpr auto HYKV_MAP_FLAGS = MAP_SHARED_VALIDATE | MAP_SYNC;
static constexpr auto HYKV_DRAM_MAP_FLAGS = MAP_ANONYMOUS | MAP_PRIVATE;
static constexpr auto HYKV_FILE_OPEN_FLAGS = O_CREAT | O_RDWR | O_DIRECT;

struct HykvConfig {
    double resize_threshold = 0.85;
    double reclaim_free_percentage = 0.4;
    size_t reclaim_threshold = 1'000'000;
    uint8_t num_recovery_threads = 64;
    size_t dax_alignment = ONE_GB;
    size_t fs_alignment = ONE_GB;
    bool enable_reclamation = false;
};

namespace internal {

template <typename K, typename V>
constexpr data_offset_size_t get_num_slots_per_page() {
    const uint32_t entry_size = sizeof(K) + sizeof(V);
    uint16_t current_page_size = PAGE_SIZE;

    const uint16_t page_overhead = sizeof(version_lock_t) + 1;
    while (entry_size + page_overhead > current_page_size && current_page_size <= BLOCK_SIZE) {
        current_page_size += PAGE_SIZE;
    }

    page_size_t num_pages_per_block = current_page_size / PAGE_SIZE;
    if (num_pages_per_block == 4 || num_pages_per_block == 5) {
        num_pages_per_block = 6;
        current_page_size = BLOCK_SIZE;
    }

    uint16_t num_slots_per_page_large = current_page_size / entry_size;
    if (num_slots_per_page_large > 255) {
        num_slots_per_page_large--;
    }
    data_offset_size_t num_slots_per_page = num_slots_per_page_large;
    while ((num_slots_per_page * entry_size) + page_overhead +
                std::ceil((double) num_slots_per_page / 8) > current_page_size) {
        num_slots_per_page--;
    }
    assert(num_slots_per_page > 0 && "Cannot fit KV pair into single page!");
    return num_slots_per_page;
}

struct VarSizeEntry {
    union {
        uint32_t size_info;
        struct {
            bool is_set : 1;
            uint16_t key_size : 15;
            uint16_t value_size : 16;
        };
    };
    char* data;

    VarSizeEntry(const uint64_t keySize, const uint64_t valueSize)
        : is_set{true}, key_size{static_cast<uint16_t>(keySize)}, value_size{static_cast<uint16_t>(valueSize)} {}
};

struct VarEntryAccessor {
    bool is_set;
    uint32_t key_size;
    uint32_t value_size;
    char* key_data = nullptr;
    char* value_data = nullptr;

    VarEntryAccessor(const char* raw_entry) {
        const VarSizeEntry* entry = reinterpret_cast<const VarSizeEntry*>(raw_entry);
        is_set = entry->is_set;
        key_size = entry->key_size;
        value_size = entry->value_size;
        key_data = const_cast<char*>(raw_entry) + sizeof(entry->size_info);
        value_data = key_data + key_size;
    }

    VarEntryAccessor(const char* raw_key_entry, const char* raw_value_entry) : VarEntryAccessor{raw_key_entry} {
        assert(value_size == 0);
        assert(key_data != nullptr);
        const VarSizeEntry* value_entry = reinterpret_cast<const VarSizeEntry*>(raw_value_entry);
        value_size = value_entry->value_size;
        value_data = const_cast<char*>(raw_value_entry) + sizeof(value_entry->size_info);
    }

    std::string_view key() {
        return std::string_view{key_data, key_size};
    }

    std::string_view value() {
        assert(value_data != nullptr);
        return std::string_view{value_data, value_size};
    }
};

template <typename K, typename V>
struct alignas(PAGE_SIZE) HykvPage {
    using VEntry = std::pair<K, V>;
    static constexpr data_offset_size_t num_slots_per_page = get_num_slots_per_page<K, V>();

    std::atomic<version_lock_t> version_lock;
    std::bitset<num_slots_per_page> free_slots;

    std::array<VEntry, num_slots_per_page> data;

    void init() {
        static constexpr size_t v_page_size = sizeof(*this);
        static_assert(((v_page_size & (v_page_size - 1)) == 0), "VPage needs to be a power of 2!");
        static_assert(PAGE_SIZE % alignof(*this) == 0, "VPage not page size conform!");
        version_lock = 0;
        free_slots.set();
    }

    inline bool lock(const bool blocking = true) {
        version_lock_t lock_value = version_lock.load(LOAD_ORDER);
        // Compare and swap until we are the thread to set the lock bit
        lock_value &= UNLOCKED_BIT;
        while (!version_lock.compare_exchange_weak(lock_value, lock_value + 1)) {
            lock_value &= UNLOCKED_BIT;
            if (!blocking) { return false; }
        }
        return true;
    }

    inline void unlock() {
        const version_lock_t current_version = version_lock.load(LOAD_ORDER);
        const version_lock_t new_version = (current_version + 1) | (current_version & CLIENT_BIT);
        version_lock.store(new_version, STORE_ORDER);
    }
};

template <>
struct alignas(PAGE_SIZE) HykvPage<std::string, std::string> {
    char* next_insert_pos;
    std::atomic<version_lock_t> version_lock;
    uint8_t modified_percentage;

    static constexpr size_t METADATA_SIZE = sizeof(version_lock) + sizeof(next_insert_pos) + sizeof(modified_percentage);
    static constexpr uint16_t DATA_SIZE = PAGE_SIZE - METADATA_SIZE;
    std::array<char, DATA_SIZE> data;

    void init() {
        static constexpr size_t v_page_size = sizeof(*this);
        static_assert(((v_page_size & (v_page_size - 1)) == 0), "VPage needs to be a power of 2!");
        static_assert(PAGE_SIZE % alignof(*this) == 0, "VPage not page size conform!");
        static_assert(PAGE_SIZE == v_page_size, "VPage not 4096 Byte!");
        version_lock = 0;
        next_insert_pos = nullptr;
        modified_percentage = 0;
    }

    inline bool lock(const bool blocking = true) {
        version_lock_t lock_value = version_lock.load(LOAD_ORDER);
        // Compare and swap until we are the thread to set the lock bit
        lock_value &= UNLOCKED_BIT;
        while (!version_lock.compare_exchange_weak(lock_value, lock_value + 1)) {
            lock_value &= UNLOCKED_BIT;
            if (!blocking) { return false; }
        }
        return true;
    }

    inline void unlock() {
        const version_lock_t current_version = version_lock.load(LOAD_ORDER);
        const version_lock_t new_version = (current_version + 1) | (current_version & CLIENT_BIT);
        version_lock.store(new_version, STORE_ORDER);
    }
};

template <typename VPage, page_size_t num_pages>
struct alignas(PAGE_SIZE) HykvPageBlock {
    /**
     * Array to store all persistent HykvPages.
     * Don't use a vector here because a HykvPage uses arrays and the whole struct would be moved on a vector resize,
     * making all pointers invalid.
     */
    std::array<VPage, num_pages> v_pages;

    inline bool is_owned() const {
        return IS_BIT_SET(v_pages[0].version_lock, CLIENT_BIT);
    }

    inline bool is_unused() const {
        return IS_BIT_SET(v_pages[0].version_lock, UNUSED_BIT);
    }
};

} // namespace internal

struct HykvFileMetadata {
    const size_t block_offset;
    const size_t block_size;
    const size_t alloc_size;
    std::atomic<block_size_t> num_used_blocks;
    block_size_t num_allocated_blocks;
    size_t total_mapped_size;
};

struct HykvFileMapping {
    const size_t mapped_size;
    void* const start_addr;
};

struct HykvBase {
    const int file_descriptor;
    const bool is_new_db;
    const bool is_file_based;
    HykvFileMetadata* const v_metadata;
    std::vector<HykvFileMapping> v_mappings;
};

struct HykvInitData {
    int fd;
    HykvFileMetadata* meta;
    std::vector<HykvFileMapping> mappings;
};

template <typename KeyT>
struct KeyAccessor {
    typedef const KeyT* checker_type;
};

template <>
struct KeyAccessor<std::string> {
    typedef std::string_view checker_type;
};

template <typename ValueT>
struct ValueAccessor {
    typedef ValueT* type;
    typedef const ValueT* const_type;
    typedef ValueT* ptr_type;
    typedef const ValueT* const_ptr_type;
    typedef const ValueT& ref_type;
    typedef const ValueT* checker_type;

    static ptr_type to_ptr_type(type& x) { return x; }
    static const_ptr_type to_ptr_type(const type& x) { return x; }
};

template <>
struct ValueAccessor<std::string> {
    typedef std::string_view type;
    typedef const std::string_view const_type;
    typedef std::string_view* ptr_type;
    typedef const std::string_view* const_ptr_type;
    typedef const std::string_view& ref_type;
    typedef const std::string_view checker_type;

    static ptr_type to_ptr_type(type& x) { return &x; }
    static const_ptr_type to_ptr_type(const type& x) { return &x; }
};

template <typename K, typename V>
class Hykv {
    using HykvT = Hykv<K, V>;
    using VPage = internal::HykvPage<K, V>;
    using KVOffset = KeyValueOffset;
    static constexpr uint64_t v_page_size = sizeof(VPage);
    static_assert(BLOCK_SIZE % v_page_size == 0, "Page needs to fit into block.");
    static constexpr page_size_t num_pages_per_block = BLOCK_SIZE / v_page_size;
    using VPageBlock = internal::HykvPageBlock<VPage, num_pages_per_block>;

  public:
    static std::unique_ptr<Hykv<K, V>> create(const std::string& pool_file, uint64_t initial_pool_size,
                                                            HykvConfig v_config = HykvConfig{});
    static std::unique_ptr<Hykv<K, V>> open(const std::string& pool_file, HykvConfig v_config = HykvConfig{});
    Hykv(HykvBase v_base, std::filesystem::path pool_dir, bool owns_pool, HykvConfig v_config);
    ~Hykv();

	void reclaim();
    void reclaim(size_t numa_node);

    class ReadOnlyClient {
        friend class Hykv<K, V>;
      public:
        bool get(const K& key, V* value) const;
		root_task get_coru(const K& key, V* value, int thread_id) const;
      protected:
        explicit ReadOnlyClient(HykvT& hykv);
        inline const std::pair<typename KeyAccessor<K>::checker_type, typename ValueAccessor<V>::checker_type> get_const_entry_from_offset(KVOffset offset) const;
        inline bool get_const_value_from_offset(KVOffset offset, V* value) const;
        HykvT& hykv_;
    };

    class Client : public ReadOnlyClient {
        friend class Hykv<K, V>;
      public:
        bool put(const K& key, const V& value);

        bool get(const K& key, V* value);
        bool get(const K& key, V* value) const;

        template <typename UpdateFn>
        bool update(const K& key, UpdateFn update_fn);

        bool remove(const K& key);

        ~Client();

      protected:
        Client(HykvT& hykv);

        bool put(const K& key, const V& value, bool delete_old);
        inline void update_access_information(size_t numa_node);
        inline void update_var_size_page_information(size_t numa_node);
        inline bool get_value_from_offset(KVOffset offset, V* value);
        inline void info_sync(bool force = false, size_t numa_node = default_numa_node_vpage);
        void free_occupied_slot(const KVOffset offset_to_delete, const K& key, const bool delete_offset = false);
        void invalidate_record(VPage* v_page, const data_offset_size_t data_offset);

        enum PageStrategy : uint8_t { BlockBased, DimmBased };

        PageStrategy strategy_;

        // Fixed size entries
        block_size_t v_block_number_;
        page_size_t v_page_number_;
        VPageBlock* v_block_;
        VPage* v_page_;

        // Dimm-based
        block_size_t end_v_block_number_;

        // Block-based
        page_size_t num_v_pages_processed_;

        uint16_t op_count_;
        size_t num_reclaimable_ops_;
        int size_delta_;
    };

    Client get_client(size_t numa_node);
	Client get_client();
    ReadOnlyClient get_read_only_client();

  protected:
    static HykvBase init_pool(const std::string& pool_file, uint64_t pool_size,
                               bool is_new_pool, HykvConfig v_config);

    void get_new_access_information(Client* client, size_t numa_node);
    void get_block_based_access(Client* client);
    void get_new_var_size_access_information(Client* client, size_t numa_node);
    KVOffset get_new_block();
    void remove_client(Client* client);

    HykvFileMapping allocate_v_page_blocks(size_t numa_node);
    void add_v_page_blocks(HykvFileMapping mapping);
    void recover_database();
    void trigger_resize(size_t numa_node);
    void trigger_reclaim(size_t num_reclaim_ops, size_t numa_node);
    void compact(Client& client, VPageBlock* v_block);

    bool check_key_equality(const K& key, const KVOffset offset_to_compare);

    HykvBase v_base_;
    const bool owns_pool_;
    HykvConfig v_config_;
    std::filesystem::path pool_dir_;

    cceh::CCEH<K> map_;
    static constexpr bool using_fp = requires_fingerprint(K);

    std::vector<VPageBlock*> v_blocks_;
    std::atomic<size_t> num_v_blocks_;
    std::atomic<size_t> current_size_;
    std::atomic<size_t> reclaimable_ops_;
    std::atomic<offset_size_t> current_block_page_;
    moodycamel::ConcurrentQueue<block_size_t> free_blocks_;

    const double resize_threshold_;
    std::atomic<bool> is_resizing_;
    std::unique_ptr<std::thread> resize_thread_;
    std::atomic<bool> is_v_blocks_resizing_;

    const size_t reclaim_threshold_;
    std::atomic<bool> is_reclaiming_;
    std::unique_ptr<std::thread> reclaim_thread_;

    std::atomic<bool> deadlock_offset_lock_;
    std::vector<KVOffset> deadlock_offsets_;

    std::atomic<uint8_t> num_active_clients_;
    const uint8_t num_recovery_threads_;
};

template <typename K, typename V>
std::unique_ptr<Hykv<K, V>> Hykv<K, V>::create(const std::string& pool_file, uint64_t initial_pool_size,
                                                 HykvConfig v_config) {
    return std::make_unique<Hykv<K, V>>(
            init_pool(pool_file, initial_pool_size, true, v_config), pool_file, true, v_config);
}

template <typename K, typename V>
std::unique_ptr<Hykv<K, V>> Hykv<K, V>::open(const std::string& pool_file, HykvConfig v_config) {
    return std::make_unique<Hykv<K, V>>(init_pool(pool_file, 0, false, v_config), pool_file, true, v_config);
}

template <typename K, typename V>
Hykv<K, V>::Hykv(HykvBase v_base, const std::filesystem::path pool_dir, const bool owns_pool, const HykvConfig v_config) :
    v_base_{v_base}, map_{131072}, owns_pool_{owns_pool}, v_config_{v_config}, pool_dir_{pool_dir},
    resize_threshold_{v_config.resize_threshold}, reclaim_threshold_{v_config.reclaim_threshold},
    num_recovery_threads_{v_config.num_recovery_threads} {
    current_block_page_ = 0;
    current_size_ = 0;
    reclaimable_ops_ = 0;
    is_resizing_ = false;
    is_reclaiming_ = false;
    num_active_clients_ = 0;
    deadlock_offset_lock_ = false;

    std::srand(std::time(nullptr));

    if (v_base_.v_mappings.empty()) {
        throw new std::runtime_error("Need to have at least one memory section mapped.");
    }

    for (HykvFileMapping mapping : v_base_.v_mappings) {
        add_v_page_blocks(mapping);
    }

    if (!v_base_.is_new_db) {
        DEBUG_LOG("Recovering existing database.");
        recover_database();
    }
    current_block_page_ = KVOffset{v_base.v_metadata->num_used_blocks.load(LOAD_ORDER), 0, 0}.offset;
}

template <typename K, typename V>
Hykv<K, V>::~Hykv() {
    if (owns_pool_) {
        DEBUG_LOG("Closing pool file.");
        //munmap(v_base_.v_metadata, v_base_.v_metadata->block_offset);
		//free(v_base_.v_metadata);
        for (const HykvFileMapping& mapping : v_base_.v_mappings) {
            //munmap(mapping.start_addr, mapping.mapped_size);
			//free(mapping.start_addr);
        }
        close(v_base_.file_descriptor);
    }
}

HykvInitData init_dram_pool(uint64_t pool_size, HykvConfig v_config, const size_t block_size, size_t numa_node) {
    const size_t alloc_size = v_config.fs_alignment;
    const size_t num_alloc_chunks = pool_size / alloc_size;
    if (num_alloc_chunks == 0) {
        throw std::runtime_error("Pool too small: " + std::to_string(pool_size));
    }

    //void* pmem_addr = mmap(nullptr, pool_size, HYKV_MAP_PROT, HYKV_DRAM_MAP_FLAGS, -1, 0);
    //MMAP_CHECK(pmem_addr)
	void *pmem_addr = (void*) numa_alloc_onnode(pool_size, numa_node);

    HykvFileMetadata v_metadata{ .block_offset = PAGE_SIZE, .block_size = block_size,
                                  .alloc_size = alloc_size, .num_used_blocks = 0,
                                  .num_allocated_blocks = 0, .total_mapped_size = pool_size };

    HykvFileMetadata* metadata = static_cast<HykvFileMetadata*>(pmem_addr);
    memcpy(metadata, &v_metadata, sizeof(v_metadata));

    char* map_start_addr = (char*) pmem_addr;
    // Each alloc chunk gets its own mapping.
    std::vector<HykvFileMapping> mappings;
    mappings.reserve(num_alloc_chunks);

    // First chunk is special because of the initial metadata chunk
    HykvFileMapping initial_mapping{ .mapped_size = alloc_size - metadata->block_offset,
                                      .start_addr = map_start_addr + metadata->block_offset };
    mappings.push_back(initial_mapping);
    block_size_t num_allocated_blocks = initial_mapping.mapped_size / block_size;

    // Remaining chunks are all the same
    for (size_t alloc_chunk = 1; alloc_chunk < num_alloc_chunks; ++alloc_chunk) {
        char* alloc_chunk_start = map_start_addr + (alloc_chunk * alloc_size);
        HykvFileMapping mapping{.mapped_size = alloc_size, .start_addr = alloc_chunk_start};
        mappings.push_back(mapping);
    }
    num_allocated_blocks += (num_alloc_chunks - 1) * (alloc_size / block_size);

    // Update num allocated blocks with correct counting of allocation chunks.
    metadata->num_allocated_blocks = num_allocated_blocks;
    pmem_persist(metadata, sizeof(HykvFileMetadata));
    return HykvInitData{ .fd = -1, .meta = metadata, .mappings = std::move(mappings) };

}

HykvInitData init_devdax_pool(const std::string& pool_file, uint64_t pool_size,
                               bool is_new_pool, HykvConfig v_config, const size_t block_size) {
    const int fd = ::open(pool_file.c_str(), O_RDWR);
    if (fd < 0) {
        IO_ERROR("Cannot open dax device: " + pool_file);
    }

    size_t alloc_size = v_config.dax_alignment;
    if (!is_new_pool) {
        const size_t meta_map_size = alloc_size;
        void* metadata_addr = mmap(nullptr, meta_map_size, PROT_READ, MAP_SHARED, fd, 0);
        if (metadata_addr == nullptr) {
            IO_ERROR("Cannot mmap pool file: " + pool_file);
        }
        const HykvFileMetadata* metadata = static_cast<HykvFileMetadata*>(metadata_addr);
        if (metadata->num_allocated_blocks > 0 && metadata->block_size > 0) {
            pool_size = metadata->total_mapped_size;
            alloc_size = metadata->alloc_size;
        } else {
            DEBUG_LOG("Opening empty database. Creating new one instead.");
            is_new_pool = true;
        }
        //munmap(metadata_addr, meta_map_size);
		//free(metadata_addr);
    }

    if (pool_size % alloc_size != 0) {
        throw std::runtime_error("Pool needs to be allocated in 1 GB chunks.");
    }

    const size_t num_alloc_chunks = pool_size / alloc_size;
    if (num_alloc_chunks == 0) {
        throw std::runtime_error("Pool too small: " + std::to_string(pool_size));
    }

    void* pmem_addr = mmap(nullptr, pool_size, HYKV_MAP_PROT, HYKV_MAP_FLAGS, fd, 0);
    //MMAP_CHECK(pmem_addr)

    if (is_new_pool) {
        HykvFileMetadata v_metadata{ .block_offset = PAGE_SIZE, .block_size = block_size,
                                      .alloc_size = alloc_size, .num_used_blocks = 0,
                                      .num_allocated_blocks = 0, .total_mapped_size = pool_size};
        pmem_memcpy_persist(pmem_addr, &v_metadata, sizeof(v_metadata));
    }
    HykvFileMetadata* metadata = static_cast<HykvFileMetadata*>(pmem_addr);
    char* map_start_addr = (char*) pmem_addr;

    // Each alloc chunk gets its own mapping.
    std::vector<HykvFileMapping> mappings;
    mappings.reserve(num_alloc_chunks);

    // First chunk is special because of the initial metadata chunk
    HykvFileMapping initial_mapping{ .mapped_size = alloc_size - metadata->block_offset,
                                      .start_addr = map_start_addr + metadata->block_offset };
    mappings.push_back(initial_mapping);
    block_size_t num_allocated_blocks = initial_mapping.mapped_size / block_size;

    // Remaining chunks are all the same
    for (size_t alloc_chunk = 1; alloc_chunk < num_alloc_chunks; ++alloc_chunk) {
        char* alloc_chunk_start = map_start_addr + (alloc_chunk * alloc_size);
        HykvFileMapping mapping{.mapped_size = alloc_size, .start_addr = alloc_chunk_start};
        mappings.push_back(mapping);
    }
    num_allocated_blocks += (num_alloc_chunks - 1) * (alloc_size / block_size);

    // Update num allocated blocks with correct counting of allocation chunks.
    metadata->num_allocated_blocks = num_allocated_blocks;
    pmem_persist(metadata, sizeof(HykvFileMetadata));
    return HykvInitData{ .fd = fd, .meta = metadata, .mappings = std::move(mappings) };
}

HykvInitData init_file_pool(const std::string& pool_dir, uint64_t pool_size,
                             bool is_new_pool, HykvConfig v_config, const size_t block_size) {
    if (is_new_pool && std::filesystem::exists(pool_dir) && !std::filesystem::is_empty(pool_dir)) {
        throw std::runtime_error("Cannot create new database in non-empty directory");
    }

    std::filesystem::create_directory(pool_dir);
    const std::filesystem::path meta_file = pool_dir + "/meta";
    HykvFileMetadata* metadata;
    const int meta_fd = ::open(meta_file.c_str(), HYKV_FILE_OPEN_FLAGS, 0644);
    if (meta_fd < 0) {
        IO_ERROR("Cannot open meta file: " + meta_file.string());
    }
    size_t alloc_size = v_config.fs_alignment;
    if (is_new_pool) {
        if (ftruncate(meta_fd, PAGE_SIZE) != 0) {
            IO_ERROR("Could not truncate: " + meta_file.string());
        }
    } else {
        const size_t meta_map_size = alloc_size;
        void *metadata_addr = mmap(nullptr, PAGE_SIZE, HYKV_MAP_PROT, HYKV_MAP_FLAGS, meta_fd, 0);
        MMAP_CHECK(metadata_addr)

        metadata = static_cast<HykvFileMetadata *>(metadata_addr);
        if (metadata->num_allocated_blocks > 0 && metadata->block_size > 0) {
            pool_size = metadata->total_mapped_size;
            alloc_size = metadata->alloc_size;
        } else {
            throw std::runtime_error("Bas existing database");
        }
    }

    if (pool_size % alloc_size != 0) {
        throw std::runtime_error("Pool needs to be allocated in 2 MB chunks.");
    }

    const size_t num_alloc_chunks = pool_size / alloc_size;
    if (num_alloc_chunks == 0) {
        throw std::runtime_error("Pool too small: " + std::to_string(pool_size));
    }

    // Each alloc chunk gets its own mapping.
    std::vector<HykvFileMapping> mappings;
    mappings.reserve(num_alloc_chunks);

    for (size_t chunk_num = 0; chunk_num < num_alloc_chunks; ++chunk_num) {
        std::filesystem::path data_file = pool_dir + "/data" + std::to_string(chunk_num);
        const int data_fd = ::open(data_file.c_str(), HYKV_FILE_OPEN_FLAGS, 0644);
        if (data_fd < 0) {
            IO_ERROR("Cannot open meta file: " + data_file.string());
        }
        if (is_new_pool) {
            if (fallocate(data_fd, 0, 0, alloc_size) != 0) {
                IO_ERROR("Could not allocate: " + data_file.string());
            }
        }

        void* pmem_addr = mmap(nullptr, alloc_size, HYKV_MAP_PROT, HYKV_MAP_FLAGS, data_fd, 0);
        MMAP_CHECK(pmem_addr)
        HykvFileMapping mapping{.mapped_size = alloc_size, .start_addr = (char*) pmem_addr};
        mappings.push_back(mapping);
        ::close(data_fd);
    }

    const size_t num_allocated_blocks = num_alloc_chunks * (alloc_size / block_size);

    if (is_new_pool) {
        void* metadata_addr = mmap(nullptr, alloc_size, HYKV_MAP_PROT, HYKV_MAP_FLAGS, meta_fd, 0);
        MMAP_CHECK(metadata_addr)
        HykvFileMetadata v_metadata{ .block_offset = PAGE_SIZE, .block_size = block_size,
                .alloc_size = alloc_size, .num_used_blocks = 0,
                .num_allocated_blocks = num_allocated_blocks, .total_mapped_size = pool_size};
        pmem_memcpy_persist(metadata_addr, &v_metadata, sizeof(v_metadata));
        metadata = static_cast<HykvFileMetadata*>(metadata_addr);
    }
    ::close(meta_fd);
    return HykvInitData{ .fd = -1, .meta = metadata, .mappings = std::move(mappings) };
}

template <typename K, typename V>
HykvBase Hykv<K, V>::init_pool(const std::string& pool_file, uint64_t pool_size,
                                 bool is_new_pool, HykvConfig v_config) {
    constexpr size_t block_size = sizeof(VPageBlock);
    HykvInitData init_data;

    const auto start = std::chrono::high_resolution_clock::now();

#ifdef HYKV_DRAM
    const bool is_file_based = false;
    init_data = init_dram_pool(pool_size, v_config, block_size, numa_node_vpage);
#else
    DEBUG_LOG((is_new_pool ? "Creating" : "Opening") << " pool file " << pool_file);
    const bool is_file_based = pool_file.find("/dev/dax") == std::string::npos;
    if (is_file_based) {
        init_data = init_file_pool(pool_file, pool_size, is_new_pool, v_config, block_size);
    } else {
        init_data = init_devdax_pool(pool_file, pool_size, is_new_pool, v_config, block_size);
    }
#endif

    const auto end = std::chrono::high_resolution_clock::now();
    DEBUG_LOG((is_new_pool ? "Creating" : "Opening") << " took " << ((end - start).count() / 1e6) << " ms");
    return HykvBase{ .file_descriptor = init_data.fd, .is_new_db = is_new_pool,
                      .is_file_based = is_file_based, .v_metadata = init_data.meta,
                      .v_mappings = std::move(init_data.mappings) };
}

template <typename K, typename V>
HykvFileMapping Hykv<K, V>::allocate_v_page_blocks(size_t numa_node) {
    const size_t alloc_size = v_base_.v_metadata->alloc_size;

    void* pmem_addr;
#ifdef HYKV_DRAM
    //pmem_addr = mmap(nullptr, alloc_size, HYKV_MAP_PROT, HYKV_DRAM_MAP_FLAGS, -1, 0);
	pmem_addr = (void*) numa_alloc_onnode(alloc_size, numa_node);
#else
    if (v_base_.is_file_based) {
        size_t next_file_id = v_base_.v_metadata->total_mapped_size / alloc_size;
        std::filesystem::path data_file = pool_dir_ / ("data" + std::to_string(next_file_id));
        DEBUG_LOG("Added data file " << data_file);
        const int data_fd = ::open(data_file.c_str(), HYKV_FILE_OPEN_FLAGS, 0644);
        if (data_fd < 0) {
            IO_ERROR("Cannot open meta file: " + data_file.string());
        }
        if (fallocate(data_fd, 0, 0, alloc_size) != 0) {
            IO_ERROR("Could not allocate: " + data_file.string());
        }
        pmem_addr = mmap(nullptr, alloc_size, HYKV_MAP_PROT, HYKV_MAP_FLAGS, data_fd, 0);
        ::close(data_fd);
    } else {
        const size_t offset = v_base_.v_metadata->total_mapped_size;
        const int fd = v_base_.file_descriptor;
        pmem_addr = mmap(nullptr, alloc_size, HYKV_MAP_PROT, HYKV_MAP_FLAGS, fd, offset);
    }
#endif

    //MMAP_CHECK(pmem_addr)
    const block_size_t num_blocks_to_map = alloc_size / sizeof(VPageBlock);
    v_base_.v_metadata->num_allocated_blocks += num_blocks_to_map;
    v_base_.v_metadata->total_mapped_size += alloc_size;
    pmem_persist(v_base_.v_metadata, sizeof(HykvFileMetadata));
    DEBUG_LOG("Allocated " << num_blocks_to_map << " blocks in " << (alloc_size / ONE_GB) << " GiB.");

    HykvFileMapping mapping{alloc_size, pmem_addr};
    v_base_.v_mappings.push_back(mapping);
    return mapping;
}

template <typename K, typename V>
void Hykv<K, V>::add_v_page_blocks(HykvFileMapping mapping) {
    VPageBlock* start_block = reinterpret_cast<VPageBlock*>(mapping.start_addr);
    const block_size_t num_blocks_to_map = mapping.mapped_size / sizeof(VPageBlock);

    is_v_blocks_resizing_.store(true, STORE_ORDER);
    v_blocks_.reserve(v_blocks_.size() + num_blocks_to_map);
    is_v_blocks_resizing_.store(false, STORE_ORDER);
    for (block_size_t block_offset = 0; block_offset < num_blocks_to_map; ++block_offset) {
        v_blocks_.push_back(start_block + block_offset);
    }
    num_v_blocks_.store(v_blocks_.size(), STORE_ORDER);
}

template <typename K, typename V>
void Hykv<K, V>::recover_database() {
    auto start = std::chrono::high_resolution_clock::now();

    const block_size_t num_used_blocks = v_base_.v_metadata->num_used_blocks.load(LOAD_ORDER);
    DEBUG_LOG("Re-inserting values from " << num_used_blocks << " blocks.");

    std::vector<std::thread> recovery_threads;
    recovery_threads.reserve(num_recovery_threads_);

    auto key_check_fn = [&](auto key, auto offset) { return check_key_equality(key, offset); };

    auto recover = [&](const size_t thread_num, const block_size_t start_block, const block_size_t end_block) {
        size_t num_entries = 0;
        for (block_size_t block_num = start_block; block_num < end_block; ++block_num) {
            VPageBlock* block = v_blocks_[block_num];
            for (page_size_t page_num = 0; page_num < num_pages_per_block; ++page_num) {
                const VPage& page = block->v_pages[page_num];
                if (IS_BIT_SET(page.version_lock, UNUSED_BIT)) {
                    // Page is empty
                    continue;
                }
                for (data_offset_size_t slot_num = 0; slot_num < VPage::num_slots_per_page; ++slot_num) {
                    if (page.free_slots[slot_num]) {
                        // No data, continue
                        continue;
                    }

                    // Data is present
                    const K& key = page.data[slot_num].first;
                    const KVOffset offset{block_num, page_num, slot_num};
                    map_.Insert(key, offset, key_check_fn);
                    num_entries++;
                }
            }
        }
        current_size_.fetch_add(num_entries);
    };

    // We give each thread + 1 blocks to avoid leaving out blocks at the end.
    const block_size_t num_blocks_per_thread = (num_used_blocks / num_recovery_threads_) + 1;

    for (size_t thread_num = 0; thread_num < num_recovery_threads_; ++thread_num) {
        const block_size_t start_block = thread_num * num_blocks_per_thread;
        const block_size_t end_block = std::min(start_block + num_blocks_per_thread, num_used_blocks);
        recovery_threads.emplace_back(recover, thread_num, start_block, end_block);
    }

    for (std::thread& thread : recovery_threads) {
        thread.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    DEBUG_LOG("RECOVERY DURATION: " << duration << " ms.");
    DEBUG_LOG("Re-inserted " << current_size_.load(LOAD_ORDER) << " keys.");
}

template <>
void Hykv<std::string, std::string>::recover_database() {
    // TODO
    throw std::runtime_error("Not implemented yet");
}

template <typename K, typename V>
void Hykv<K, V>::get_new_access_information(Client* client, size_t numa_node) {
    // Get insert/delete count info
    client->info_sync(true, numa_node);

    const KVOffset block_page{current_block_page_.load(LOAD_ORDER)};
    if (block_page.block_number > resize_threshold_ * v_blocks_.size()) {
        trigger_resize(numa_node);
    }

    get_block_based_access(client);
}

template <typename K, typename V>
void Hykv<K, V>::get_new_var_size_access_information(Client* client, size_t numa_node) {
    if constexpr (!std::is_same_v<K, std::string>) {
        throw std::runtime_error("Cannot update var pages for fixed-size entries.");
    }

    get_new_access_information(client, numa_node);
    client->v_page_->next_insert_pos = client->v_page_->data.data();

    v_base_.v_metadata->num_used_blocks.fetch_add(1, std::memory_order_relaxed);
    pmem_persist(v_base_.v_metadata, sizeof(HykvFileMetadata));
}

template <typename K, typename V>
void Hykv<K, V>::get_block_based_access(Client* client) {
    block_size_t client_block = -1;
    page_size_t client_page = 0;
    if (!free_blocks_.try_dequeue(client_block)) {
        // No free block available, get new one.
        const KVOffset new_block = get_new_block();
        client_block = new_block.block_number;
        client_page = new_block.page_number;
    }
    assert(client_block != -1);

    client->strategy_ = Client::PageStrategy::BlockBased;
    client->v_block_number_ = client_block;
    client->v_page_number_ = client_page;
    client->num_v_pages_processed_ = 0;

    while (is_v_blocks_resizing_.load(std::memory_order_acquire)) {
        // Wait for vector's memmove to complete. Otherwise we might encounter a segfault.
        asm("nop");
    }

    if (client->v_block_ != nullptr) {
        client->v_block_->v_pages[0].version_lock &= NO_CLIENT_BIT;
    }

    client->v_block_ = v_blocks_[client_block];
    client->v_page_ = &(client->v_block_->v_pages[client_page]);
    client->v_page_->init();
    client->v_block_->v_pages[0].version_lock |= CLIENT_BIT;

    v_base_.v_metadata->num_used_blocks.fetch_add(1, std::memory_order_relaxed);
    pmem_persist(v_base_.v_metadata, sizeof(HykvFileMetadata));
}

template <typename K, typename V>
KeyValueOffset Hykv<K, V>::get_new_block() {
    offset_size_t raw_block_page = current_block_page_.load(LOAD_ORDER);
    KVOffset new_offset{};
    block_size_t client_block;
    do {
        const KVOffset v_block_page{raw_block_page};
        client_block = v_block_page.block_number;

        const block_size_t new_block = client_block + 1;
        while (client_block >= num_v_blocks_.load(LOAD_ORDER)) {
            asm("nop");
        }
        assert(new_block < v_blocks_.size());

        // Choose random offset to evenly distribute load on all DIMMs
        page_size_t new_page = 0;
        if constexpr (!std::is_same_v<std::string, K>) {
            new_page = rand() % num_pages_per_block;
        }
        new_offset = KVOffset{new_block, new_page, 0};
    } while (!current_block_page_.compare_exchange_weak(raw_block_page, new_offset.offset));

    return KVOffset{raw_block_page};
}

template <typename K, typename V>
void Hykv<K, V>::trigger_resize(size_t numa_node) {
    bool expected_resizing = false;
    const bool should_resize = is_resizing_.compare_exchange_strong(expected_resizing, true);
    if (!should_resize) {
        return;
    }

    // Only one thread can ever get here because for all others the atomic exchange above fails.
    resize_thread_ = std::make_unique<std::thread>([this, numa_node] {
        DEBUG_LOG("Start resizing.");
        // Used for benchmarks. remove.
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(0, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
        HykvFileMapping mapping = allocate_v_page_blocks(numa_node);
        add_v_page_blocks(mapping);
        is_resizing_.store(false, STORE_ORDER);
        DEBUG_LOG("End resizing.");
    });
    resize_thread_->detach();
}

template <typename K, typename V>
void Hykv<K, V>::trigger_reclaim(size_t num_reclaim_ops, size_t numa_node) {
    bool expected_reclaiming = false;
    const bool should_reclaim = is_reclaiming_.compare_exchange_strong(expected_reclaiming, true);
    if (!should_reclaim) {
        return;
    }

    reclaimable_ops_.fetch_sub(num_reclaim_ops);

    reclaim_thread_ = std::make_unique<std::thread>([this, numa_node] {
        DEBUG_LOG("START RECLAIMING...");
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(71, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
        reclaim(numa_node);
        is_reclaiming_.store(false, STORE_ORDER);
        DEBUG_LOG("END RECLAIMING");
    });
    reclaim_thread_->detach();
}


template <typename K, typename V>
inline typename Hykv<K, V>::Client Hykv<K, V>::get_client(size_t numa_node) {
    num_active_clients_++;
    Client client{*this};
    if constexpr (std::is_same_v<K, std::string>) {
        get_new_var_size_access_information(&client, numa_node);
    } else {
        get_new_access_information(&client, numa_node);
    }
    return client;
}

template <typename K, typename V>
inline typename Hykv<K, V>::Client Hykv<K, V>::get_client() {
    num_active_clients_++;
    Client client{*this};
    if constexpr (std::is_same_v<K, std::string>) {
        get_new_var_size_access_information(&client, default_numa_node_vpage);
    } else {
        get_new_access_information(&client, default_numa_node_vpage);
    }
    return client;
}

template <typename K, typename V>
typename Hykv<K, V>::ReadOnlyClient Hykv<K, V>::get_read_only_client() {
    return ReadOnlyClient{*this};
}

template <typename K, typename V>
void Hykv<K, V>::remove_client(Hykv::Client* client) {
    client->info_sync(true, default_numa_node_vpage);
    --num_active_clients_;
}

template <typename K, typename V>
inline bool Hykv<K, V>::check_key_equality(const K& key, const KVOffset offset_to_compare) {
    if constexpr (!using_fp) {
        throw std::runtime_error("Should not use key checker without fingerprints!");
    }

    if (offset_to_compare.is_tombstone()) {
        return false;
    }

    const ReadOnlyClient client = get_read_only_client();
    const auto& entry = client.get_const_entry_from_offset(offset_to_compare);
    if constexpr (std::is_pointer_v<typename KeyAccessor<K>::checker_type>) {
        return *(entry.first) == key;
    } else {
        return entry.first == key;
    }
}

template <typename K, typename V>
bool Hykv<K, V>::Client::put(const K& key, const V& value, bool delete_old) {
	// Enter here, 2nd
	// printf("Insert--hykv-test1\n");
    v_page_->lock();

	//Decide the data placement of DRAM or PMem
	size_t numa_node = numa_node_vpage;

    // We now have the lock on this page
    std::bitset<VPage::num_slots_per_page>* free_slots = &v_page_->free_slots;
    const data_offset_size_t free_slot_idx = free_slots->_Find_first();

    if (free_slot_idx >= free_slots->size()) {
        // Page is full. Free lock on page and restart.
        v_page_->unlock();
        update_access_information(numa_node);
        return put(key, value, delete_old);
    }

	// cache line flushing
#ifdef CACHE_LINE_FLUSHING
		size_t key_hash = hykv::cceh::h(&key, sizeof(key));
		std::string key_s = std::to_string(key_hash);
		char const *key_c = key_s.c_str();
		String key_ = String(key_c);
		bool not_flush = checkData (key_, true);
#endif

    // We have found a free slot on this page. Persist data.
	// i.e., store data in PMem
	v_page_->data[free_slot_idx] = {key, value};
#ifdef CACHE_LINE_FLUSHING	
	if(not_flush){
#endif		
		// sizeof(typename VPage::VEntry) = 216
		typename VPage::VEntry* entry_ptr = v_page_->data.data() + free_slot_idx;
    	pmem_persist(entry_ptr, sizeof(typename VPage::VEntry));
#ifdef CACHE_LINE_FLUSHING		
	}
#endif	

	free_slots->reset(free_slot_idx);
#ifdef CACHE_LINE_FLUSHING		
	if(not_flush){
#endif		
		// sizeof(*free_slots) = 8
    	pmem_persist(free_slots, sizeof(*free_slots));
#ifdef CACHE_LINE_FLUSHING		
	}
#endif

    // Store data in DRAM offset map.
    const KVOffset kv_offset{v_block_number_, v_page_number_, free_slot_idx};
    KVOffset old_offset;

    if constexpr (using_fp) {
		// Enter here
        auto key_check_fn = [&](auto key, auto offset) { return this->hykv_.check_key_equality(key, offset); };
        old_offset = this->hykv_.map_.Insert(key, kv_offset, key_check_fn);
    } else {
        old_offset = this->hykv_.map_.Insert(key, kv_offset);
    }

    const bool is_new_item = old_offset.is_tombstone();
    if (!is_new_item && delete_old) {
        // Need to free slot at old location for this key
        free_occupied_slot(old_offset, key);
    }

    v_page_->unlock();

    // We have added one value, so +1
    size_delta_++;
    info_sync(false, numa_node);

    return is_new_item;
}

template <>
bool Hykv<std::string, std::string>::Client::put(const std::string& key, const std::string& value, const bool delete_old) {
	// Not enter here
	//printf("Insert--hykv-test2\n");
    v_page_->lock();
    VPage* start_v_page = v_page_;

	//Decide the data placement of DRAM or PMem
	size_t numa_node = numa_node_vpage;

    internal::VarSizeEntry entry{key.size(), value.size()};
    bool is_inserted = false;
    char* insert_pos = v_page_->next_insert_pos;
    const size_t entry_length = entry.key_size + entry.value_size;
    const size_t meta_size = sizeof(entry.size_info);

    ptrdiff_t offset_in_page = insert_pos - reinterpret_cast<char*>(v_page_);
    assert(offset_in_page <= v_page_size);

    block_size_t current_block_number = v_block_number_;
    page_size_t current_page_number = v_page_number_;
    VPage* current_v_page = v_page_;

    if (offset_in_page + meta_size + entry_length > v_page_size) {
        // This page is not big enough to fit the record. Allocate new one.
        update_var_size_page_information(numa_node);

        // Writing to new block. Treat this as a fresh insert.
        const bool force_new_page = v_block_number_ != current_block_number;

        if (offset_in_page + meta_size + entry.key_size <= v_page_size && !force_new_page) {
            // Key fits, value does not. Add key to old page and value to new one.
            char* key_insert_pos = insert_pos;

            // Add value to new page.
            // 0 size indicates key is on previous page.
            insert_pos = v_page_->data.data();
            internal::VarSizeEntry value_entry{0, value.size()};
            value_entry.data = insert_pos + meta_size;
            memcpy(value_entry.data, value.data(), value_entry.value_size);
            memcpy(insert_pos, &value_entry.size_info, meta_size);
            pmem_persist(insert_pos, meta_size + value_entry.value_size);

            // 0 size indicates value is on next page.
            entry.value_size = 0;
            entry.data = key_insert_pos + meta_size;
            memcpy(entry.data, key.data(), entry.key_size);
            memcpy(key_insert_pos, &entry.size_info, meta_size);
            pmem_persist(key_insert_pos, meta_size + entry.key_size);

            current_v_page->next_insert_pos = reinterpret_cast<char*>(current_v_page);
            v_page_->next_insert_pos = insert_pos + meta_size + value_entry.value_size;
            current_v_page = v_page_;
            is_inserted = true;
        } else {
            // Both key and value don't fit. Add both to new page. 0 sizes indicate that both are on next page.
            if (offset_in_page + meta_size <= v_page_size) {
                // 0-entry metadata fits into current page.
                internal::VarSizeEntry next_page_entry{0, 0};
                pmem_memcpy_persist(insert_pos, &next_page_entry.size_info, meta_size);
            }

            current_v_page->next_insert_pos = reinterpret_cast<char*>(current_v_page);
            insert_pos = v_page_->data.data();
            offset_in_page = v_page_->METADATA_SIZE;
            current_block_number = v_block_number_;
            current_page_number = v_page_number_;
            current_v_page = v_page_;
        }
    }

    if (!is_inserted) {
        // Entire record fits into current page.
        entry.data = insert_pos + meta_size;
        memcpy(entry.data, key.data(), entry.key_size);
        memcpy(entry.data + entry.key_size, value.data(), entry.value_size);
        pmem_persist(entry.data, entry_length);
        memcpy(insert_pos, &entry.size_info, meta_size);
        pmem_persist(insert_pos, meta_size);
        v_page_->next_insert_pos = insert_pos + (meta_size + entry_length);
    }

    const uint16_t data_offset = offset_in_page - v_page_->METADATA_SIZE;
    const KVOffset var_offset{current_block_number, current_page_number, data_offset};
    bool is_new_item = true;
    KVOffset old_offset = KVOffset::Tombstone();

    // Store data in DRAM map.
    auto key_check_fn = [&](auto key, auto offset) { return this->hykv_.check_key_equality(key, offset); };
    old_offset = this->hykv_.map_.Insert(key, var_offset, key_check_fn);
    is_new_item = old_offset.is_tombstone();
    size_delta_++;

    start_v_page->unlock();

    // Need to free slot at old location for this key
    if (!is_new_item && delete_old) {
        free_occupied_slot(old_offset, key);
    }

    info_sync(false, numa_node);
    return is_new_item;
}

template <typename K, typename V>
bool Hykv<K, V>::Client::put(const K& key, const V& value) {
    return put(key, value, true);
}

template <typename K, typename V>
bool Hykv<K, V>::Client::get(const K& key, V* value) {
	// not enter here
	//printf("get--hykv-test1\n");
    auto key_check_fn = [&](auto key, auto offset) {
        if constexpr (using_fp) { return this->hykv_.check_key_equality(key, offset); }
        else { return cceh::CCEH<K>::dummy_key_check(key, offset); }
    };

    while (true) {
        KVOffset kv_offset = this->hykv_.map_.Get(key, key_check_fn);
        if (kv_offset.is_tombstone()) {
            return false;
        }
        if (get_value_from_offset(kv_offset, value)) {
            return true;
        }
    }
}

template <typename K, typename V>
bool Hykv<K, V>::ReadOnlyClient::get(const K& key, V* value) const {
	// enter here, 2nd
	//printf("get--hykv-test2\n");
    auto key_check_fn = [&](auto key, auto offset) {
        if constexpr (using_fp) { 
			//enter here
			//printf("using_fp\n");
			return this->hykv_.check_key_equality(key, offset); 
		}
        else { return cceh::CCEH<K>::dummy_key_check(key, offset); }
    };

    while (true) {
		// Most time-consuming part
		//Get the kv_offset from the offset map (DRAM)
		KVOffset kv_offset = this->hykv_.map_.Get(key, key_check_fn);
        if (kv_offset.is_tombstone()) {
            return false;
        }
		//Get the value from the vblock (PMem)
        if (get_const_value_from_offset(kv_offset, value)) {
            return true;
        }
    }
}

// Coroutine
template <typename K, typename V>
root_task Hykv<K, V>::ReadOnlyClient::get_coru(const K& key, V* value, int thread_id) const {
	// enter here
	//printf("get--hykv-test2\n");
	//printf("get_coru: %d\n", thread_id);
    auto key_check_fn = [&](auto key, auto offset) {
        if constexpr (using_fp) { return this->hykv_.check_key_equality(key, offset); }
        else { return cceh::CCEH<K>::dummy_key_check(key, offset); }
    };

    while (true) {
		// Most time-consuming part
		//coroutine
		KVOffset kv_offset_ref = this->hykv_.map_.Get(key, key_check_fn);
		KVOffset kv_offset = co_await prefetch(kv_offset_ref, thread_id);

		//KVOffset kv_offset = this->hykv_.map_.Get(key, key_check_fn);
        if (kv_offset.is_tombstone()) {
            //return false;
			co_return false;
        }
        if (get_const_value_from_offset(kv_offset, value)) {
            //return true;
			co_return true;
        }
    }
}



template <typename K, typename V>
bool Hykv<K, V>::Client::get(const K& key, V* value) const {
    return static_cast<const Hykv<K, V>::ReadOnlyClient*>(this)->get(key, value);
}

template <typename K, typename V>
template <typename UpdateFn>
bool Hykv<K, V>::Client::update(const K& key, UpdateFn update_fn) {
    if constexpr (std::is_same_v<K, std::string>) {
        throw std::runtime_error("In-place update not supported for variable length records!");
    }

    auto key_check_fn = [&](auto key, auto offset) {
        if constexpr (using_fp) { return this->hykv_.check_key_equality(key, offset); }
        else { return cceh::CCEH<K>::dummy_key_check(key, offset); }
    };

    while (true) {
        const KVOffset kv_offset = this->hykv_.map_.Get(key, key_check_fn);
        if (kv_offset.is_tombstone()) {
            return false;
        }

        const auto [block, page, slot] = kv_offset.get_offsets();
        VPage& v_page = this->hykv_.v_blocks_[block]->v_pages[page];
        if (!v_page.lock(false)) {
            // Could not lock page, so the record could be modified and we need to try again
            continue;
        }

        update_fn(&(v_page.data[slot].second));
        v_page.unlock();
        return true;
    }
}

template <typename K, typename V>
bool Hykv<K, V>::Client::remove(const K& key) {
    auto key_check_fn = [&](auto key, auto offset) {
        if constexpr (using_fp) { return this->hykv_.check_key_equality(key, offset); }
        else { return cceh::CCEH<K>::dummy_key_check(key, offset); }
    };

    const KVOffset kv_offset = this->hykv_.map_.Get(key, key_check_fn);
    if (kv_offset.is_tombstone()) {
        return false;
    }

    free_occupied_slot(kv_offset, key, true);
    num_reclaimable_ops_++;
    return true;
}

template <typename K, typename V>
void Hykv<K, V>::Client::free_occupied_slot(const KVOffset offset_to_delete, const K& key, const bool delete_offset) {
    const auto [block_number, page_number, data_offset] = offset_to_delete.get_offsets();
    while (this->hykv_.is_v_blocks_resizing_.load(LOAD_ORDER)) {
        // Wait for vector's memmove to complete. Otherwise we might encounter a segfault.
    }

    auto key_check_fn = [&](auto key, auto offset) {
        if constexpr (using_fp) { return this->hykv_.check_key_equality(key, offset); }
        else { return cceh::CCEH<K>::dummy_key_check(key, offset); }
    };

    if (v_block_number_ == block_number && v_page_number_ == page_number) {
        // Old record to delete is on the same page. We already hold the lock here.
        invalidate_record(v_page_, data_offset);
        if (delete_offset) {
            this->hykv_.map_.Insert(key, IndexV::NONE(), key_check_fn);
        }
        --size_delta_;
        return;
    }

    VPage& v_page = this->hykv_.v_blocks_[block_number]->v_pages[page_number];
    std::atomic<version_lock_t>& v_lock = v_page.version_lock;
    version_lock_t lock_value = v_lock.load(LOAD_ORDER);
    lock_value &= UNLOCKED_BIT;

    const size_t num_deadlock_retries = 32;
    std::vector<KVOffset>& deadlock_offsets = this->hykv_.deadlock_offsets_;
    bool deadlock_offset_inserted = false;
    bool has_lock;
    // Acquire lock or clear deadlock loop
    while (true) {
        size_t retries = 0;
        has_lock = true;
        while (!v_lock.compare_exchange_weak(lock_value, lock_value + 1)) {
            // Client may be in a deadlock situation
            lock_value &= UNLOCKED_BIT;
            if (++retries == num_deadlock_retries) {
                has_lock = false;
                break;
            }
            _mm_pause();
        }

        if (has_lock) {
            // Acquired lock, delete normally
            invalidate_record(&v_page, data_offset);
            break;
        }

        // Check possible deadlock
        bool expected_offset_lock = false;
        while (!this->hykv_.deadlock_offset_lock_.compare_exchange_weak(expected_offset_lock, true)) {
            expected_offset_lock = false;
            _mm_pause();
        }

        if (!deadlock_offset_inserted) {
            deadlock_offsets.push_back(offset_to_delete);
            deadlock_offset_inserted = true;
        }

        std::vector<KVOffset> new_offsets;
        new_offsets.reserve(deadlock_offsets.size());
        bool encountered_own_offset = false;
        for (KVOffset offset : deadlock_offsets) {
            encountered_own_offset |= offset_to_delete == offset;
            if (offset.block_number != v_block_number_ || offset.page_number != v_page_number_) {
                new_offsets.push_back(offset);
            } else {
                invalidate_record(v_page_, offset.data_offset);
            }
        }
        deadlock_offsets = std::move(new_offsets);
        this->hykv_.deadlock_offset_lock_.store(false);

        if (!encountered_own_offset) {
            // The old offset that this client need to delete was deleted by another client, we can return.
            break;
        }
    }

    if (delete_offset) {
        this->hykv_.map_.Insert(key, IndexV::NONE(), key_check_fn);
    }

    if (has_lock) {
        if (deadlock_offset_inserted) {
            // We inserted into the queue but manually deleted.
            bool expected_offset_lock = false;
            while (!this->hykv_.deadlock_offset_lock_.compare_exchange_weak(expected_offset_lock, true)) {
                expected_offset_lock = false;
                _mm_pause();
            }

            auto own_item_pos = std::find(deadlock_offsets.begin(), deadlock_offsets.end(), offset_to_delete);
            assert(own_item_pos != deadlock_offsets.end());
            deadlock_offsets.erase(own_item_pos);
            this->hykv_.deadlock_offset_lock_.store(false);
        }
        v_page.unlock();
    }

    --size_delta_;
}

template <typename K, typename V>
inline void Hykv<K, V>::Client::invalidate_record(VPage* v_page, const data_offset_size_t data_offset) {
    if constexpr (std::is_same_v<K, std::string>) {
        char* raw_data = &v_page->data[data_offset];
        internal::VarSizeEntry* var_entry = reinterpret_cast<internal::VarSizeEntry*>(raw_data);
        var_entry->is_set = false;
        const size_t meta_size = sizeof(var_entry->size_info);
        pmem_persist(&var_entry->size_info, meta_size);
        const size_t entry_size = var_entry->key_size + var_entry->value_size + meta_size;
        v_page->modified_percentage += (entry_size * 100) / VPage::DATA_SIZE;
    } else {
        auto* free_slots = &v_page->free_slots;
        free_slots->set(data_offset);
        pmem_persist(free_slots, sizeof(*free_slots));
    }
}

template <typename K, typename V>
void Hykv<K, V>::Client::update_access_information(size_t numa_node) {
    if (strategy_ == PageStrategy::DimmBased) {
        if (v_block_number_ == end_v_block_number_) {
            // No more allocated pages, need new range
            this->hykv_.get_new_access_information(this, numa_node);
        } else {
            ++v_block_number_;
            v_block_ = this->hykv_.v_blocks_[v_block_number_];
            v_page_ = &(v_block_->v_pages[v_page_number_]);
        }
    } else if (strategy_ == PageStrategy::BlockBased) {
        if (++num_v_pages_processed_ == this->hykv_.num_pages_per_block) {
            // No more pages, need new block
            this->hykv_.get_new_access_information(this, numa_node);
        } else {
            v_page_number_ = (v_page_number_ + 1) % this->hykv_.num_pages_per_block;
            v_page_ = &(v_block_->v_pages[v_page_number_]);
        }
    } else {
        throw std::runtime_error("Unknown page strategy.");
    }

    // Make sure new page is initialized correctly
    v_page_->init();
}

template <typename K, typename V>
void Hykv<K, V>::Client::update_var_size_page_information(size_t numa_node) {
    update_access_information(numa_node);
    v_page_->next_insert_pos = v_page_->data.data();
}

template <typename K, typename V>
void Hykv<K, V>::Client::info_sync(const bool force, size_t numa_node) {
    if (force || ++op_count_ == 10000) {
        this->hykv_.current_size_.fetch_add(size_delta_);

        if (this->hykv_.v_config_.enable_reclamation) {
            size_t num_reclaims = this->hykv_.reclaimable_ops_.fetch_add(num_reclaimable_ops_);
            num_reclaims += num_reclaimable_ops_;
            if (num_reclaims > this->hykv_.reclaim_threshold_) {
                this->hykv_.trigger_reclaim(num_reclaims, numa_node);
            }
        }

        op_count_ = 0;
        size_delta_ = 0;
        num_reclaimable_ops_ = 0;
    }
}

template <typename K, typename V>
Hykv<K, V>::ReadOnlyClient::ReadOnlyClient(HykvT& hykv) : hykv_{hykv} {}

template <typename K, typename V>
Hykv<K, V>::Client::Client(HykvT& hykv) : ReadOnlyClient{hykv} {
    op_count_ = 0;
    size_delta_ = 0;
    num_v_pages_processed_ = 0;
    v_block_number_ = 0;
    v_page_number_ = 0;
    end_v_block_number_ = 0;
    v_page_ = nullptr;
    v_block_ = nullptr;
}

template <typename K, typename V>
Hykv<K, V>::Client::~Client() {
    this->hykv_.remove_client(this);
    if (v_block_ != nullptr) {
        v_block_->v_pages[0].version_lock &= NO_CLIENT_BIT;
    }
}

template <typename K, typename V>
inline const std::pair<typename KeyAccessor<K>::checker_type, typename ValueAccessor<V>::checker_type>
Hykv<K, V>::ReadOnlyClient::get_const_entry_from_offset(Hykv::KVOffset offset) const {
    if constexpr (std::is_same_v<K, std::string>) {
        const auto[block, page, data_offset] = offset.get_offsets();
        const VPageBlock* v_block = this->hykv_.v_blocks_[block];
        const VPage& v_page = v_block->v_pages[page];
        const char* raw_data = &v_page.data[data_offset];
        internal::VarEntryAccessor var_entry{raw_data};
        if (var_entry.value_size == 0) {
            // Value is on next page
            const char* raw_value_data = &v_block->v_pages[page + 1].data[0];
            var_entry = internal::VarEntryAccessor{raw_data, raw_value_data};
        }
        return {var_entry.key(), var_entry.value()};
    } else {
        const auto[block, page, slot] = offset.get_offsets();
        const auto& entry = this->hykv_.v_blocks_[block]->v_pages[page].data[slot];
        return {&entry.first, &entry.second};
    }
}

template <typename K, typename V>
inline bool Hykv<K, V>::ReadOnlyClient::get_const_value_from_offset(KVOffset offset, V* value) const {
    const auto [block, page, slot] = offset.get_offsets();
    const VPage& v_page = this->hykv_.v_blocks_[block]->v_pages[page];
    const std::atomic<version_lock_t>& page_lock = v_page.version_lock;
    version_lock_t lock_val = page_lock.load(LOAD_ORDER);
    if (IS_LOCKED(lock_val)) {
        return false;
    }
    const auto entry = this->get_const_entry_from_offset(offset);
    if constexpr (std::is_same_v<K, std::string>) {
        const std::string_view& str_val = entry.second;
        value->assign(str_val.data(), str_val.size());
    } else {
        *value = *(entry.second);
    }
    return lock_val == page_lock.load(LOAD_ORDER);
}

template <typename K, typename V>
inline bool Hykv<K, V>::Client::get_value_from_offset(const KVOffset offset, V* value) {
    const auto [block, page, slot] = offset.get_offsets();
    const VPage& v_page = this->hykv_.v_blocks_[block]->v_pages[page];
    const std::atomic<version_lock_t>& page_lock = v_page.version_lock;
    version_lock_t lock_val = page_lock.load(LOAD_ORDER);
    if (IS_LOCKED(lock_val)) {
        return false;
    }
    *value = v_page.data[slot].second;
    return lock_val == page_lock.load(LOAD_ORDER);
}

template <>
inline bool Hykv<std::string, std::string>::Client::get_value_from_offset(const KVOffset offset, std::string* value) {
    const auto [block, page, data_offset] = offset.get_offsets();
    const VPageBlock* v_block = this->hykv_.v_blocks_[block];
    const VPage& v_page = v_block->v_pages[page];
    const std::atomic<version_lock_t>& page_lock = v_page.version_lock;
    version_lock_t lock_val = page_lock.load(LOAD_ORDER);
    if (IS_LOCKED(lock_val)) {
        return false;
    }

    const char *raw_data = &v_page.data[data_offset];
    internal::VarEntryAccessor var_entry{raw_data};
    if (var_entry.value_size == 0) {
        // Value is on next page
        const char *raw_value_data = &v_block->v_pages[page + 1].data[0];
        var_entry = internal::VarEntryAccessor{raw_data, raw_value_data};
    }
    value->assign(var_entry.value_data, var_entry.value_size);
    return lock_val == page_lock.load(LOAD_ORDER);
}

template <typename K, typename V>
void Hykv<K, V>::compact(Client& client, VPageBlock* v_block) {
    for (VPage& v_page : v_block->v_pages) {
        v_page.lock();
        auto& free_slots = v_page.free_slots;
        for (size_t slot = 0; slot < v_page.num_slots_per_page; ++slot) {
            if (free_slots[slot]) {
                // Slot is free
                continue;
            }

            const auto& record = v_page.data[slot];
            client.put(record.first, record.second, false);
            free_slots[slot] = 1;
            pmem_persist(&v_page.free_slots, sizeof(v_page.free_slots));
        }
        v_page.unlock();
    }

}

template <>
void Hykv<std::string, std::string>::compact(Client& client, VPageBlock* v_block) {
    const size_t meta_size = sizeof(internal::VarSizeEntry::size_info);

    page_size_t current_page = 0;
    VPage* v_page = &v_block->v_pages[current_page];
    v_page->lock();
    const char* raw_data = v_page->data.data();
    const char* next_insert_pos = v_page->next_insert_pos;
    bool is_last_page = reinterpret_cast<const void*>(next_insert_pos) != reinterpret_cast<const void*>(v_page);

    auto key_check_fn = [&](auto key, auto offset) { return check_key_equality(key, offset); };

    while (true) {
        internal::VarEntryAccessor var_entry{raw_data};
        data_offset_size_t data_skip = meta_size + var_entry.key_size + var_entry.value_size;
        const ptrdiff_t offset_in_page = raw_data - reinterpret_cast<char*>(v_page);
        const bool end_of_page = (offset_in_page + meta_size) > v_page_size;

        if (is_last_page && raw_data == next_insert_pos) {
            // Previous client has not written further than this, so we can stop here.
            return;
        }

        // No more entries in this page
        if (end_of_page || (var_entry.key_size == 0 && var_entry.value_size == 0)) {
            if (++current_page == num_pages_per_block) {
                return;
            }

            v_page->unlock();
            v_page = &v_block->v_pages[current_page];
            v_page->lock();
            next_insert_pos = v_page->next_insert_pos;
            raw_data = v_page->data.data();
            is_last_page = reinterpret_cast<const void*>(next_insert_pos) != reinterpret_cast<const void*>(v_page);
            continue;
        }

        // Value is on next page
        if (var_entry.value_size == 0) {
            // Value is on next page
            if (++current_page == num_pages_per_block) {
                return;
            }

            v_page->unlock();
            v_page = &v_block->v_pages[current_page];
            v_page->lock();
            next_insert_pos = v_page->next_insert_pos;
            is_last_page = reinterpret_cast<const void*>(next_insert_pos) != reinterpret_cast<const void*>(v_page);
            const char* raw_value_data = v_page->data.data();
            var_entry = internal::VarEntryAccessor{raw_data, raw_value_data};
            raw_data = raw_value_data + meta_size + var_entry.value_size;
            data_skip = 0;
        }

        // Insert record into new block
        if (var_entry.is_set) {
            const std::string key{var_entry.key()};
            IndexV offset = map_.Get(key, key_check_fn);
            if (!offset.is_tombstone()) {
                client.put(key, std::string{var_entry.value()}, false);
                var_entry.is_set = false;
                pmem_persist(&var_entry.is_set, sizeof(var_entry.is_set));
            }

        }
        raw_data += data_skip;
    }

    v_page->unlock();
}

template <typename K, typename V>
void Hykv<K, V>::reclaim() {
    const size_t num_slots_per_block = num_pages_per_block * VPage::num_slots_per_page;
    const block_size_t max_block = KVOffset{current_block_page_.load(LOAD_ORDER)}.block_number;

    // At least X percent of the block should be free before reclaiming it.
    const size_t free_threshold = v_config_.reclaim_free_percentage * num_slots_per_block;
    size_t total_freed_blocks = 0;
    Client client = get_client(default_numa_node_vpage);

    for (block_size_t block_num = 0; block_num < max_block; ++block_num) {
        VPageBlock* v_block = v_blocks_[block_num];
        size_t block_free_slots = 0;
        if (v_block->is_owned() || v_block->is_unused()) {
            // Block in use by client or already marked as free.
            continue;
        }

        for (const VPage& v_page : v_block->v_pages) {
            block_free_slots += v_page.free_slots.count();
        }

        if (block_free_slots > free_threshold) {
            compact(client, v_block);
            VPage& head_page = v_block->v_pages[0];
            head_page.version_lock = UNUSED_BIT;
            free_blocks_.enqueue(block_num);
            total_freed_blocks++;
        }
    }

    DEBUG_LOG("TOTAL FREED BLOCKS: " << total_freed_blocks);
}

template <typename K, typename V>
void Hykv<K, V>::reclaim(size_t numa_node) {
    const size_t num_slots_per_block = num_pages_per_block * VPage::num_slots_per_page;
    const block_size_t max_block = KVOffset{current_block_page_.load(LOAD_ORDER)}.block_number;

    // At least X percent of the block should be free before reclaiming it.
    const size_t free_threshold = v_config_.reclaim_free_percentage * num_slots_per_block;
    size_t total_freed_blocks = 0;
    Client client = get_client(numa_node);

    for (block_size_t block_num = 0; block_num < max_block; ++block_num) {
        VPageBlock* v_block = v_blocks_[block_num];
        size_t block_free_slots = 0;
        if (v_block->is_owned() || v_block->is_unused()) {
            // Block in use by client or already marked as free.
            continue;
        }

        for (const VPage& v_page : v_block->v_pages) {
            block_free_slots += v_page.free_slots.count();
        }

        if (block_free_slots > free_threshold) {
            compact(client, v_block);
            VPage& head_page = v_block->v_pages[0];
            head_page.version_lock = UNUSED_BIT;
            free_blocks_.enqueue(block_num);
            total_freed_blocks++;
        }
    }

    DEBUG_LOG("TOTAL FREED BLOCKS: " << total_freed_blocks);
}

template <>
void Hykv<std::string, std::string>::reclaim() {
    const block_size_t max_block = KVOffset{current_block_page_.load(LOAD_ORDER)}.block_number;
    const double modified_threshold = v_config_.reclaim_free_percentage;
    size_t total_freed_blocks = 0;
    Client client = get_client(default_numa_node_vpage);

    for (block_size_t block_num = 0; block_num < max_block; ++block_num) {
        VPageBlock* v_block = v_blocks_[block_num];
        if (v_block->is_owned() || v_block->is_unused()) {
            // Block in use by client or already marked as free.
            continue;
        }

        double modified_percentage = 0;
        for (const VPage& v_page : v_block->v_pages) {
            modified_percentage += ((double)v_page.modified_percentage / num_pages_per_block) / 100;
            if (modified_percentage > modified_threshold) {
                compact(client, v_block);
                VPage& head_page = v_block->v_pages[0];
                head_page.version_lock = UNUSED_BIT;
                free_blocks_.enqueue(block_num);
                total_freed_blocks++;
                break;
            }

            if (v_page.next_insert_pos != reinterpret_cast<const char*>(&v_page)) {
                // This page has not been completed, so no need to check next one.
                break;
            }
        }
    }

    DEBUG_LOG("TOTAL FREED BLOCKS: " << total_freed_blocks);
}

template <>
void Hykv<std::string, std::string>::reclaim(size_t numa_node) {
    const block_size_t max_block = KVOffset{current_block_page_.load(LOAD_ORDER)}.block_number;
    const double modified_threshold = v_config_.reclaim_free_percentage;
    size_t total_freed_blocks = 0;
    Client client = get_client(numa_node);

    for (block_size_t block_num = 0; block_num < max_block; ++block_num) {
        VPageBlock* v_block = v_blocks_[block_num];
        if (v_block->is_owned() || v_block->is_unused()) {
            // Block in use by client or already marked as free.
            continue;
        }

        double modified_percentage = 0;
        for (const VPage& v_page : v_block->v_pages) {
            modified_percentage += ((double)v_page.modified_percentage / num_pages_per_block) / 100;
            if (modified_percentage > modified_threshold) {
                compact(client, v_block);
                VPage& head_page = v_block->v_pages[0];
                head_page.version_lock = UNUSED_BIT;
                free_blocks_.enqueue(block_num);
                total_freed_blocks++;
                break;
            }

            if (v_page.next_insert_pos != reinterpret_cast<const char*>(&v_page)) {
                // This page has not been completed, so no need to check next one.
                break;
            }
        }
    }

    DEBUG_LOG("TOTAL FREED BLOCKS: " << total_freed_blocks);
}

}  // namespace hykv
