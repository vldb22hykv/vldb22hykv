#pragma once

#include "../benchmark.hpp"
#include "common_fixture.hpp"
#include "hykv/hykv.hpp"

#include "hykv/coru_infra.hpp"
#include "hykv/sclf.hpp"

namespace hykv {
namespace kv_bm {

template <typename KeyT = KeyType16, typename ValueT = ValueType200>
class HykvFixture : public BaseFixture {
  public:
    typedef KeyT KeyType;
    using HykvT = Hykv<KeyT, ValueT>;

    void InitMap(const uint64_t num_prefill_inserts = 0, const bool re_init = true) override;
    void InitMap(const uint64_t num_prefill_inserts, HykvConfig v_config);

    void DeInitMap() override;

    uint64_t insert(uint64_t start_idx, uint64_t end_idx) final;

    uint64_t setup_and_insert(uint64_t start_idx, uint64_t end_idx) final;
    uint64_t setup_and_find(uint64_t start_idx, uint64_t end_idx, uint64_t num_finds) final;
    uint64_t setup_and_delete(uint64_t start_idx, uint64_t end_idx, uint64_t num_deletes) final;
    uint64_t setup_and_update(uint64_t start_idx, uint64_t end_idx, uint64_t num_updates) final;

    uint64_t setup_and_get_update(uint64_t start_idx, uint64_t end_idx, uint64_t num_updates);

    uint64_t run_ycsb(uint64_t start_idx, uint64_t end_idx,
        const std::vector<ycsb::Record>& data, hdr_histogram* hdr) final;

    HykvT* getHykv() {
        return hykv_.get();
    }

  protected:
    std::unique_ptr<HykvT> hykv_;
    bool hykv_initialized_ = false;
    std::string pool_file_;
};

template <typename KeyT, typename ValueT>
void HykvFixture<KeyT, ValueT>::InitMap(uint64_t num_prefill_inserts, const bool re_init) {
    if (hykv_initialized_ && !re_init) {
        return;
    }

    return InitMap(num_prefill_inserts, HykvConfig{});
}

template <typename KeyT, typename ValueT>
void HykvFixture<KeyT, ValueT>::InitMap(uint64_t num_prefill_inserts, HykvConfig v_config) {
#ifdef CCEH_PERSISTENT
    PMemAllocator::get().initialize();
#endif

    pool_file_ = HYKV_POOL_FILE;
//    pool_file_ = random_file(DB_PMEM_DIR);
//    pool_file_ = DB_PMEM_DIR + std::string("/hykv");

//    hykv_ = HykvT::open(pool_file_, v_config);
    hykv_ = HykvT::create(pool_file_, BM_POOL_SIZE, v_config);
    this->prefill(num_prefill_inserts);
    hykv_initialized_ = true;
}

template <typename KeyT, typename ValueT>
void HykvFixture<KeyT, ValueT>::DeInitMap() {
    BaseFixture::DeInitMap();
    hykv_ = nullptr;
    hykv_initialized_ = false;
    if (pool_file_.find("/dev/dax") == std::string::npos) {
        std::filesystem::remove_all(pool_file_);
    }
}

//#ifdef DATA_MIGRATION	
unsigned long int kvs_counter = 0;
bool kvs_counter_flag = true;
//#endif 

template <typename KeyT, typename ValueT>
uint64_t HykvFixture<KeyT, ValueT>::insert(uint64_t start_idx, uint64_t end_idx) {
	// Enter here, 1st
	// printf("Insert-test1\n");
    uint64_t insert_counter = 0;
    auto v_client = hykv_->get_client();

    for (uint64_t key = start_idx; key < end_idx; ++key) {
		/*
		kvs_counter++;
		if (kvs_counter % 50000000 == 0){
			printf("kvs_counter: %ld\n", kvs_counter);
		}
		*/
		
#ifdef DATA_MIGRATION		
		if (kvs_counter == 200000000 && kvs_counter_flag){
			printf("DATA_MIGRATION\n");
			kvs_counter_flag = false;
			hykv::cceh::change_numa();
			numa_set_preferred(2);
		}

		if (kvs_counter > 200000000 && kvs_counter % 10000000 == 0){
			//hykv::default_numa_node_offset = 2;
			//hykv::numa_node_offset = 2;
			hykv::cceh::change_numa();
			numa_set_preferred(2);
		}

#endif
        const KeyT db_key{key};
        const ValueT value{key};
        insert_counter += v_client.put(db_key, value);
    }

    return insert_counter;
}

template <>
uint64_t HykvFixture<std::string, std::string>::insert(uint64_t start_idx, uint64_t end_idx) {
	//printf("Insert-test2\n");
    uint64_t insert_counter = 0;
    auto v_client = hykv_->get_client();
    const std::vector<std::string>& keys = std::get<0>(var_size_kvs_);
    const std::vector<std::string>& values = std::get<1>(var_size_kvs_);
    for (uint64_t key = start_idx; key < end_idx; ++key) {
        const std::string& db_key = keys[key];
        const std::string& value = values[key];
        insert_counter += v_client.put(db_key, value);
    }
    return insert_counter;
}

template <typename KeyT, typename ValueT>
uint64_t HykvFixture<KeyT, ValueT>::setup_and_insert(uint64_t start_idx, uint64_t end_idx) {
    return insert(start_idx, end_idx);
}

template <typename KeyT, typename ValueT>
uint64_t HykvFixture<KeyT, ValueT>::setup_and_find(uint64_t start_idx, uint64_t end_idx, uint64_t num_finds) {
	//Enter here, 1st
#ifdef COROUTINE
		auto thread_map = coru_map.find(pthread_self());
		//long unsigned int thread_key = thread_map->first;
		int thread_id = thread_map->second;
		//printf("thread_key: %ld, thread_id: %d\n", thread_key, thread_id);
#endif

	//printf("Get-test1\n");
    std::random_device rnd{};
    auto rnd_engine = std::default_random_engine(rnd());
    std::uniform_int_distribution<> distrib(start_idx, end_idx);

    const auto v_client = hykv_->get_read_only_client();

	//Coroutine put (write) part:
	auto v_client_write = hykv_->get_client();
	uint64_t insert_counter = 0;

    uint64_t found_counter = 0;
    ValueT value;

	//Coroutine
    for (uint64_t i = 0; i < num_finds; ++i) {
        const uint64_t key = distrib(rnd_engine);
        const KeyT db_key{key};
		// Most time-consuming part
		//Coroutine
#ifdef COROUTINE		
		//printf("for-loop, thread_key: %ld, thread_id: %d\n", thread_key, thread_id);
		throttlers[thread_id].spawn_coru(v_client.get_coru(db_key, &value, thread_id), thread_id);
#else		
		v_client.get(db_key, &value);
#endif		
		found_counter++;
        //const bool found = v_client.get(db_key, &value);
        //found_counter += found && (value == ValueT{key});

		// Test for write in the get()
		/*
		if (i%10 == 0){
        	v_client_write.put(db_key, value);
			insert_counter++;
		}
		*/
    }
	//printf ("read_counter: %ld\n", found_counter);
	//printf ("insert_counter: %ld\n", insert_counter);
    return found_counter;
}

template <>
uint64_t HykvFixture<std::string, std::string>::setup_and_find(uint64_t start_idx, uint64_t end_idx, uint64_t num_finds) {
	//printf("Get-test2\n");
    std::random_device rnd{};
    auto rnd_engine = std::default_random_engine(rnd());
    std::uniform_int_distribution<> distrib(start_idx, end_idx);

    const std::vector<std::string>& keys = std::get<0>(var_size_kvs_);
    const std::vector<std::string>& values = std::get<1>(var_size_kvs_);

    auto v_client = hykv_->get_read_only_client();
    uint64_t found_counter = 0;
    std::string result;
    for (uint64_t i = 0; i < num_finds; ++i) {
        const uint64_t key = distrib(rnd_engine);
        const std::string& db_key = keys[key];
        const std::string& value = values[key];
        const bool found = v_client.get(db_key, &result);
        found_counter += found && (result == value);
    }
    return found_counter;
}

template <typename KeyT, typename ValueT>
uint64_t HykvFixture<KeyT, ValueT>::setup_and_delete(uint64_t start_idx, uint64_t end_idx, uint64_t num_deletes) {
	// Enter here
	//printf("Delete-test1\n");
    std::random_device rnd{};
    auto rnd_engine = std::default_random_engine(rnd());
    std::uniform_int_distribution<> distrib(start_idx, end_idx);

    auto v_client = hykv_->get_client();
    uint64_t delete_counter = 0;
    for (uint64_t i = 0; i < num_deletes; ++i) {
        const uint64_t key = distrib(rnd_engine);
        const KeyT db_key{key};
        delete_counter += v_client.remove(db_key);
    }
    return delete_counter;
}

template <>
uint64_t HykvFixture<std::string, std::string>::setup_and_delete(uint64_t start_idx, uint64_t end_idx, uint64_t num_deletes) {
	//printf("Delete-test2\n");
    std::random_device rnd{};
    auto rnd_engine = std::default_random_engine(rnd());
    std::uniform_int_distribution<> distrib(start_idx, end_idx);
    const std::vector<std::string>& keys = std::get<0>(var_size_kvs_);

    auto v_client = hykv_->get_client();
    uint64_t delete_counter = 0;
    for (uint64_t i = 0; i < num_deletes; ++i) {
        const uint64_t key = distrib(rnd_engine);
        const std::string& db_key = keys[key];
        delete_counter += v_client.remove(db_key);
    }
    return delete_counter;
}

template <typename KeyT, typename ValueT>
uint64_t HykvFixture<KeyT, ValueT>::setup_and_update(uint64_t start_idx, uint64_t end_idx, uint64_t num_updates) {
	//Enter here
	//printf("Update-test1\n");
    std::random_device rnd{};
    auto rnd_engine = std::default_random_engine(rnd());
    std::uniform_int_distribution<> distrib(start_idx, end_idx);

    auto v_client = hykv_->get_client();
    uint64_t update_counter = 0;

    auto update_fn = [](ValueT* value) {
        value->update_value();
        pmem_persist(value, sizeof(uint64_t));
    };

    for (uint64_t i = 0; i < num_updates; ++i) {
        const uint64_t key = distrib(rnd_engine);
        const KeyT db_key{key};
        update_counter += v_client.update(db_key, update_fn);
    }
    return update_counter;
}

template <>
uint64_t HykvFixture<std::string, std::string>::setup_and_update(uint64_t start_idx, uint64_t end_idx, uint64_t num_updates) {
    throw std::runtime_error("Not supported");
}

template <typename KeyT, typename ValueT>
uint64_t HykvFixture<KeyT, ValueT>::setup_and_get_update(uint64_t start_idx, uint64_t end_idx, uint64_t num_updates) {
	//printf("Update-test2\n");
    std::random_device rnd{};
    auto rnd_engine = std::default_random_engine(rnd());
    std::uniform_int_distribution<> distrib(start_idx, end_idx);

    auto v_client = hykv_->get_client();

    ValueT new_v{};
    for (uint64_t i = 0; i < num_updates; ++i) {
        const uint64_t key = distrib(rnd_engine);
        const KeyT db_key{key};
        v_client.get(db_key, &new_v);
        new_v.update_value();
        v_client.put(db_key, new_v);
    }

    return num_updates;
}

template <>
uint64_t HykvFixture<std::string, std::string>::setup_and_get_update(uint64_t, uint64_t, uint64_t) {
    throw std::runtime_error("Not supported");
}

template <typename KeyT, typename ValueT>
uint64_t HykvFixture<KeyT, ValueT>::run_ycsb(uint64_t, uint64_t, const std::vector<ycsb::Record>&, hdr_histogram*) {
    throw std::runtime_error{"YCSB not implemented for non-ycsb key/value types."};
}

template <>
uint64_t HykvFixture<KeyType8, ValueType200>::run_ycsb(
    uint64_t start_idx, uint64_t end_idx, const std::vector<ycsb::Record>& data, hdr_histogram* hdr) {
    uint64_t op_count = 0;
    auto v_client = hykv_->get_client();
    ValueType200 value;
    const ValueType200 null_value{0ul};

    std::chrono::high_resolution_clock::time_point start;
    for (int op_num = start_idx; op_num < end_idx; ++op_num) {
        const ycsb::Record& record = data[op_num];

        if (hdr != nullptr) {
            start = std::chrono::high_resolution_clock::now();
        }

        switch (record.op) {
            case ycsb::Record::Op::INSERT: {
                v_client.put(record.key, record.value);
                op_count++;
                break;
            }
            case ycsb::Record::Op::GET: {
                const bool found = v_client.get(record.key, &value);
                op_count += found && (value != null_value);
                break;
            }
            case ycsb::Record::Op::UPDATE: {
                auto update_fn = [&](ValueType200* value) {
                    value->data[0] = record.value.data[0];
                    pmem_persist(value->data.data(), sizeof(uint64_t));
                };
                op_count += v_client.update(record.key, update_fn);
                break;
            }
            default: {
                throw std::runtime_error("Unknown operation: " + std::to_string(record.op));
            }
        }


        if (hdr == nullptr) {
            continue;
        }

        const auto end = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        hdr_record_value(hdr, duration.count());
    }

    return op_count;
}

}  // namespace kv_bm
}  // namespace hykv
