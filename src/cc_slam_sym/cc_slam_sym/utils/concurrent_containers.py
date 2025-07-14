#!/usr/bin/env python3
"""
Concurrent data structures for multi-threaded SLAM system
Following GLIM's asynchronous architecture patterns
"""

import threading
from typing import List, Optional, TypeVar, Generic, Callable
from collections import deque
import time

T = TypeVar('T')


class ConcurrentVector(Generic[T]):
    """Thread-safe vector/list with blocking and non-blocking operations
    
    Similar to GLIM's concurrent_vector implementation
    """
    
    def __init__(self, max_size: Optional[int] = None):
        """Initialize concurrent vector
        
        Args:
            max_size: Maximum size of the vector (None for unlimited)
        """
        self._data = deque()
        self._max_size = max_size
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
        
    def push_back(self, item: T, block: bool = True, timeout: Optional[float] = None) -> bool:
        """Add item to the back of the vector
        
        Args:
            item: Item to add
            block: Whether to block if full
            timeout: Maximum time to wait
            
        Returns:
            True if successful, False if timeout/full
        """
        with self._lock:
            if self._max_size is not None:
                while len(self._data) >= self._max_size:
                    if not block:
                        return False
                    if not self._not_full.wait(timeout):
                        return False
                        
            self._data.append(item)
            self._not_empty.notify()
            return True
            
    def pop_front(self, block: bool = True, timeout: Optional[float] = None) -> Optional[T]:
        """Remove and return item from front
        
        Args:
            block: Whether to block if empty
            timeout: Maximum time to wait
            
        Returns:
            Item or None if timeout/empty
        """
        with self._lock:
            while len(self._data) == 0:
                if not block:
                    return None
                if not self._not_empty.wait(timeout):
                    return None
                    
            item = self._data.popleft()
            self._not_full.notify()
            return item
            
    def try_pop_front(self) -> Optional[T]:
        """Non-blocking pop from front"""
        return self.pop_front(block=False)
        
    def clear(self) -> None:
        """Clear all items"""
        with self._lock:
            self._data.clear()
            self._not_full.notify_all()
            
    def size(self) -> int:
        """Get current size"""
        with self._lock:
            return len(self._data)
            
    def empty(self) -> bool:
        """Check if empty"""
        with self._lock:
            return len(self._data) == 0
            
    def workload(self) -> int:
        """Get workload (size) for monitoring"""
        return self.size()


class AsyncWorker(Generic[T]):
    """Base class for asynchronous workers processing items from a queue
    
    Similar to GLIM's async processing pattern
    """
    
    def __init__(self, name: str, max_queue_size: Optional[int] = None):
        """Initialize async worker
        
        Args:
            name: Worker name for logging
            max_queue_size: Maximum queue size
        """
        self.name = name
        self.input_queue = ConcurrentVector[T](max_queue_size)
        self._running = False
        self._thread = None
        self._process_callback = None
        
    def set_process_callback(self, callback: Callable[[T], None]) -> None:
        """Set the processing callback
        
        Args:
            callback: Function to process each item
        """
        self._process_callback = callback
        
    def start(self) -> None:
        """Start the worker thread"""
        if self._running:
            return
            
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, name=self.name)
        self._thread.start()
        
    def stop(self) -> None:
        """Stop the worker thread"""
        self._running = False
        if self._thread:
            self._thread.join()
            
    def submit(self, item: T) -> bool:
        """Submit item for processing
        
        Args:
            item: Item to process
            
        Returns:
            True if submitted successfully
        """
        return self.input_queue.push_back(item, block=False)
        
    def workload(self) -> int:
        """Get current workload"""
        return self.input_queue.workload()
        
    def _worker_loop(self) -> None:
        """Main worker loop"""
        while self._running:
            # Get item with timeout to check running status
            item = self.input_queue.pop_front(timeout=0.1)
            
            if item is not None and self._process_callback:
                try:
                    self._process_callback(item)
                except Exception as e:
                    print(f"[{self.name}] Error processing item: {e}")


class CallbackList:
    """Thread-safe callback list
    
    Similar to GLIM's callback system
    """
    
    def __init__(self):
        """Initialize callback list"""
        self._callbacks = []
        self._lock = threading.Lock()
        
    def add(self, callback: Callable, name: Optional[str] = None) -> str:
        """Add a callback
        
        Args:
            callback: Callback function
            name: Optional name for the callback
            
        Returns:
            Callback ID/name
        """
        with self._lock:
            if name is None:
                name = f"callback_{len(self._callbacks)}"
            self._callbacks.append((name, callback))
            return name
            
    def remove(self, name: str) -> bool:
        """Remove a callback by name
        
        Args:
            name: Callback name
            
        Returns:
            True if removed
        """
        with self._lock:
            for i, (cb_name, _) in enumerate(self._callbacks):
                if cb_name == name:
                    del self._callbacks[i]
                    return True
            return False
            
    def call(self, *args, **kwargs) -> None:
        """Call all callbacks
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        # Copy callbacks to avoid holding lock during execution
        with self._lock:
            callbacks = self._callbacks.copy()
            
        for name, callback in callbacks:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                print(f"[CallbackList] Error in callback '{name}': {e}")
                
    def clear(self) -> None:
        """Clear all callbacks"""
        with self._lock:
            self._callbacks.clear()
            
    def size(self) -> int:
        """Get number of callbacks"""
        with self._lock:
            return len(self._callbacks)


class ThreadPool:
    """Simple thread pool for parallel processing
    
    Used for batch operations like in GLIM
    """
    
    def __init__(self, num_threads: int = 4):
        """Initialize thread pool
        
        Args:
            num_threads: Number of worker threads
        """
        self.num_threads = num_threads
        self._task_queue = ConcurrentVector[Callable]()
        self._threads = []
        self._running = False
        
    def start(self) -> None:
        """Start the thread pool"""
        if self._running:
            return
            
        self._running = True
        for i in range(self.num_threads):
            thread = threading.Thread(target=self._worker, name=f"ThreadPool-{i}")
            thread.start()
            self._threads.append(thread)
            
    def stop(self) -> None:
        """Stop the thread pool"""
        self._running = False
        for thread in self._threads:
            thread.join()
        self._threads.clear()
        
    def submit(self, task: Callable) -> bool:
        """Submit a task
        
        Args:
            task: Task to execute
            
        Returns:
            True if submitted
        """
        return self._task_queue.push_back(task, block=False)
        
    def _worker(self) -> None:
        """Worker thread function"""
        while self._running:
            task = self._task_queue.pop_front(timeout=0.1)
            if task:
                try:
                    task()
                except Exception as e:
                    print(f"[ThreadPool] Task error: {e}")