#pragma once
#include <thread>
#include <atomic>
#include <chrono>
#include <string>

#include "stringkey.hpp"
#include "lru.hpp"
#include "chksum.hpp"
#include "libpmem.h"

#define MAX_NUM_DATA 16
#define CACHE_LINE 64

typedef ThreadSafeStringKey String;
typedef String::HashCompare HashCompare;
typedef ThreadSafeLRUCache<String, bool, HashCompare> LRUCache;
std::unique_ptr<LRUCache> cache_;
bool init_ = false;
std::thread bg_thread_;
std::atomic<bool> enable_bg_;
std::string data_base_;
// Store all checksums
// TODO: optimize its data layout
std::vector<std::vector<uint32_t>> chksum_storage_;

void chkSum(String key_);
bool EnableBG();
void SetEnableBG(bool status);
void chkSumTask();

void sclf_start(unsigned long int capacity) {
    cache_ = std::unique_ptr<LRUCache>(new LRUCache(capacity));
    SetEnableBG(true);
    bg_thread_ = std::thread(chkSumTask);
    init_ = true;
}

void sclf_stop() {
    SetEnableBG(false);
    bg_thread_.join();
}
// dirty = true: write, notify to call persist
// dirty = false: read, do nothing 
bool checkData(String key_, bool dirty) {
    return cache_->insert(key_, dirty);
}

// call this API from the application when checkData() returns true
void persist(void* src, size_t len) {
  pmem_flush(src, len);
  pmem_drain();
}

// Get data from backend-store
std::string  GetData(String key) {
    // Call backend-store API here to get the data
    std::string data = "Get data from backend-store";
    return data;
}

// Accumulate data and perform a checksum operation after obtaining enough data
// x   x   x   x chk
// x   x   x   x chk
// x   x   x   x chk
// x   x   x   x chk
// chk chk chk chk
void chkSum(String key_) {
    data_base_ += GetData(key_);   
    size_t data_base_len_ = data_base_.size();
    if (data_base_len_ >= CACHE_LINE * MAX_NUM_DATA) {
        // If the data size is not a multiple of the cache line size, add extra zeros at the end
        int addZeros = 0;
        if ((addZeros = data_base_len_ % CACHE_LINE) != 0) {
            for (int i = 0; i < addZeros; i++) {
                data_base_ += '0';
            }
            data_base_len_ += addZeros;
        }

        // Perform horizontal checksum
        std::vector<uint32_t> new_chksum_set;
        for (size_t pos = 0; pos < data_base_len_; pos += CACHE_LINE) {
            std::string data_sub_ = data_base_.substr (pos, CACHE_LINE);
            char* ptr = data_sub_.data();
            uint32_t new_chksum = avx2_sumbytes_variant2(reinterpret_cast<uint8_t*>(ptr), data_base_len_);
            new_chksum_set.emplace_back(std::move(new_chksum));
        }
         persist(new_chksum_set.data(), data_base_len_);
        // Merge new checksums into the global checksum
        chksum_storage_.emplace_back(std::move(new_chksum_set));

        // Perform vertical checksum
        // TODO
        data_base_.clear();
    }
}

bool EnableBG() { 
    return enable_bg_.load(std::memory_order_acquire); 
}

void SetEnableBG(bool status) {
    enable_bg_.store(status, std::memory_order_release);
}


void chkSumTask() {
    while (EnableBG()) {
        while (!cache_->checkEvicitItems()) {
            String key_ = cache_->getEvicitItems();
            if (key_.size() > 0) { 
                chkSum(key_);
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}