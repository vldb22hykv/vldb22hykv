#pragma once

#include <xmmintrin.h>
#include <coroutine>
#include <stdint.h>
#include "omp.h"
#include <map>
#include <pthread.h>
#include <iostream>

inline std::map<long unsigned int, int> coru_map;

struct scheduler_queue {
  static constexpr const int N = 256;
  using coro_handle = std::coroutine_handle<>;

  uint32_t head = 0;
  uint32_t tail = 0;
  coro_handle arr[N];

  void push_back(coro_handle h) {
    arr[head] = h;
    head = (head + 1) % N; 
  }

  coro_handle pop_front() {
    auto result = arr[tail];
    tail = (tail + 1) % N;
    return result;
  }
  auto try_pop_front() { return head != tail ? pop_front() : coro_handle{}; }

  void run() {
    while (auto h = try_pop_front())
      h.resume();
  }
};

//inline scheduler_queue scheduler;
inline scheduler_queue* scheduler;

// prefetch Awaitable
template <typename T> struct prefetch_Awaitable {
  T &value;
  int thread_id;

  prefetch_Awaitable(T &value, int thread_id) : value(value), thread_id(thread_id) {}

  bool await_ready() { return false; }
  T &await_resume() { return value; }
  template <typename Handle> auto await_suspend(Handle h) {
    _mm_prefetch(reinterpret_cast<char const *>(std::addressof(value)),
                 _MM_HINT_NTA);
	auto &q = scheduler[thread_id];
	q.push_back(h);
	return q.pop_front();
  }
};

template <typename T> auto prefetch(T &value, int thread_id) {
  return prefetch_Awaitable<T>{value, thread_id};
}

// Simple thread caching allocator.
struct tcalloc {
  struct header {
    header *next;
    size_t size;
  };
  header *root = nullptr;
  size_t last_size_allocated = 0;
  size_t total = 0;
  size_t alloc_count = 0;

  ~tcalloc() {
    auto current = root;
    while (current) {
      auto next = current->next;
      ::free(current);
      current = next;
    }
  }

  void *alloc(size_t sz) {
    if (root && root->size <= sz) {
      void *mem = root;
      root = root->next;
      return mem;
    }
    ++alloc_count;
    total += sz;
    last_size_allocated = sz;
    return malloc(sz);
  }

  void stats() {
    printf("allocs %zu total %zu sz %zu\n", alloc_count, total, last_size_allocated);
  }

  void free(void *p, size_t sz) {
    auto new_entry = static_cast<header *>(p);
    new_entry->size = sz;
    new_entry->next = root;
    root = new_entry;
  }
};

inline tcalloc* allocator;

struct throttler;

struct root_task {
  struct promise_type;
  using HDL = std::coroutine_handle<promise_type>;

  struct promise_type {
    throttler *owner = nullptr;
	bool value_;

    void *operator new(size_t sz){ 
		auto thread_id = coru_map.find(pthread_self());
		return allocator[thread_id->second].alloc(sz); 
	}
    void operator delete(void *p, size_t sz){ 
		auto thread_id = coru_map.find(pthread_self());
		allocator[thread_id->second].free(p, sz); 
	}
    root_task get_return_object() { return root_task{*this}; }
    std::suspend_always initial_suspend() { return {}; }
    //void return_void();
    void unhandled_exception() noexcept { std::terminate(); }
    std::suspend_never final_suspend() noexcept { return {}; }
	void return_value(bool value);
  };

  // TODO: this can be done via a wrapper coroutine
  auto set_owner(throttler *owner) {
    auto result = h;
    h.promise().owner = owner;
    h = nullptr;
	return h;
  }

  auto get_handle(throttler *owner){
	  h.promise().owner = owner;
	  return h;
  }

  ~root_task() {
    if (h)
      h.destroy();
  }

  root_task(root_task&& rhs) : h(rhs.h) { rhs.h = nullptr; }
  root_task(root_task const&) = delete;

private:
  root_task(promise_type &p) : h(HDL::from_promise(p)) {}

  HDL h;
};

inline unsigned* limits;

struct throttler {
  int throttler_id;

  static void init_throttler(size_t limit, size_t num_threads){
	scheduler = (scheduler_queue*) malloc (sizeof(scheduler_queue) * num_threads);
	for (int i = 0; i< num_threads; i++) {
		scheduler[i].head = 0;
		scheduler[i].tail = 0;
	}

	allocator = (tcalloc*) malloc(sizeof (tcalloc) * num_threads);
	for (int i = 0; i< num_threads; i++) {
		allocator[i].last_size_allocated = 0;
		allocator[i].total = 0;
		allocator[i].alloc_count = 0;
		allocator[i].root = nullptr;
	}

	limits = (unsigned*) malloc(sizeof (unsigned) * num_threads);
	for (int i = 0; i< num_threads; i++) {
		limits[i] = limit;
	}

  }

   static void set_map(std::map<long unsigned int, int> map){
	   coru_map = map;
	   /*
	   printf("set_map\n");
		for (std::map<long unsigned int, int>::iterator itr = coru_map.begin(); itr != coru_map.end(); ++itr) {
			std::cout << '\t' << itr->first
				<< '\t' << itr->second << '\n';
		}
		*/
   }

  void on_task_done() { 
	  limits[throttler_id] = limits[throttler_id] - 1;
  }

  bool spawn_coru(root_task t, int thread_id) {
		if (limits[thread_id] <= 0)
			scheduler[thread_id].pop_front().resume();
		auto h = t.get_handle(this);
		scheduler[thread_id].push_back(h);
		limits[thread_id] = limits[thread_id] + 1;	
		//auto &promise = my_h.promise();
		//return promise.value_;
		return true;
  }

  void run() {
    scheduler[throttler_id].run();
  }

  ~throttler() { run(); }
};

//void root_task::promise_type::return_void() { owner->on_task_done(); }
void root_task::promise_type::return_value(bool value) { 
  value_ = value;
  owner->on_task_done(); 
  return;
}

